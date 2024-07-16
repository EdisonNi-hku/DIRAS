import argparse
import pandas as pd
import random
import pickle
import re

from transformers import AutoTokenizer, set_seed, logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from get_training_data import SYSTEM, TRAIN_PROMPT


def find_first_float(text):
    # Regex pattern to find floating point numbers (including those with signs)
    pattern = r'[-+]?\d*\.\d+'

    # Using search to find the first occurrence
    match = re.search(pattern, text)

    if match:
        # Returning the matched float, converting to float type
        return float(match.group())
    else:
        return None


def parse_texts(outputs):
    none_count = 0
    guesses = []
    confidences = []
    none_position = []
    for i, output in enumerate(outputs):
        response = output
        if "Guess" not in response or "Confidence" not in response:
            none_count += 1
            guess = random.choice(['Yes', 'No'])
            confidence = 0
            guesses.append(guess)
            confidences.append(confidence)
            continue
        else:
            guess_text = response.split("Guess")[-1].split("Confidence")[0]
            guess = 'Yes' if 'yes' in guess_text.lower() else 'No'
            confidence_text = response.split("Confidence")[-1]
            confidence = find_first_float(confidence_text)

        if guess is None or confidence is None:
            none_count += 1
            guess = random.choice(['Yes', 'No'])
            confidence = 0
            none_position.append(i)
        guesses.append(guess)
        confidences.append(confidence)
    print("None count", none_count)
    return guesses, confidences, none_position


def parse_scores(outputs, answer_key='Guess'):
    parsed_outputs = []
    for o in outputs:
        parsed_o = []
        for tok in o:
            first_k = None
            for k, v in tok.items():
                if v['rank'] == 1:
                    first_k = k
                    break
            p = np.exp(tok[first_k]['logprob'])
            t = tok[first_k]['decoded_token']
            parsed_o.append((t, p))
        parsed_outputs.append(parsed_o)
    outputs = parsed_outputs
    none_count = 0
    guesses = []
    confidences = []
    none_position = []
    for i, scores in enumerate(outputs):
        guess = None
        confidence = None
        cumulative_text = ""
        see_answer = False
        for score in scores:
            cumulative_text += score[0]
            if answer_key in score[0] or answer_key in cumulative_text:
                see_answer = True
            if see_answer and ('yes' in score[0].lower() or 'no' in score[0].lower()):
                guess = 'Yes' if 'yes' in score[0].lower() else 'No'
                confidence = score[1]
                break
        if guess is None or confidence is None:
            none_count += 1
            guess = random.choice(['Yes', 'No'])
            confidence = 0
            none_position.append(i)
        guesses.append(guess)
        confidences.append(confidence)
    print("None count", none_count)
    return guesses, confidences, none_position


def format_prompt(tokenizer, system, input, no_system=False):
    if no_system:
        chat = [
            {"role": "user", "content": system + '\n\n' + input},
        ]
    else:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    formatted_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return formatted_input


def batchify_list(input_list, batch_size):
    # Calculate the number of batches required
    num_batches = (len(input_list) + batch_size - 1) // batch_size

    # Create empty list to hold batches
    batches = []

    # Generate batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(input_list))
        batch = input_list[start_idx:end_idx]
        batches.append(batch)

    return batches


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_dir", type=str, default="cache")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--architecture", type=str, default="llama-3")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-*b-hf")
    parser.add_argument("--question_file", type=str, default="example/question.jsonl")
    parser.add_argument("--document_file", type=str, default="example/document.jsonl")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--output_file", type=str, default="")
    parser.add_argument("--max_new_token", type=int, default=512)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    parser.add_argument("--logprobs", type=int, default=5)
    return parser.parse_args()


def main(args):
    if args.load_tokenizer:
        tokenizer_dir = args.model_path if args.lora_path is None else args.lora_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, cache_dir=args.model_cache_dir, padding_side='left', local_files_only=False)
        tokenizer_name = args.model_path
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.model_cache_dir, padding_side='left', local_files_only=False)
        tokenizer_name = args.tokenizer_path

    sys = SYSTEM
    df = pd.read_json(args.question_file, lines=True)
    df_docs = pd.read_json(args.document_file, lines=True)
    instructions = []

    # Annotate all (question, document) pairs.
    for q in df['question']:
        for d in df_docs['document']:
            explanation = df.loc[df['question'] == q, 'explanation']
            instructions.append(TRAIN_PROMPT.format(background_information=explanation, question=q, paragraph_chunk=d))

    if 'gemma' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p, no_system=True) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<end_of_turn>')]
    elif 'llama-3' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif 'phi3' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|end|>')]
    else:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id]

    if args.sample:
        prompts = random.sample(prompts, args.sample)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        logprobs=args.logprobs,
        max_tokens=args.max_new_token,
        stop_token_ids=eos_token_ids,
    )
    llm = LLM(
        model=args.model_path,
        enable_lora=True if args.lora_path is not None else False,
        download_dir=args.model_cache_dir,
        tokenizer=tokenizer_name,
        dtype='auto',
        seed=args.seed,
        trust_remote_code=True,
        max_lora_rank=64,
    )

    if args.lora_path:
        outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("lora", 1, args.lora_path))
    else:
        outputs = llm.generate(prompts, sampling_params)

    generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)
    output_df = pd.DataFrame({'prompt': prompts, 'output': generated_text})

    logprobs = []
    for o in outputs:
        all_token_probs = []
        for alternatives in o.outputs[0].logprobs:
            alternative_probs = {}
            for k, v in alternatives.items():
                alternative_probs[k] = {'logprob': v.logprob, 'rank': v.rank, 'decoded_token': v.decoded_token}
            all_token_probs.append(alternative_probs)
        logprobs.append(all_token_probs)
    with open(args.output_file.replace('.csv', '.pkl'), 'wb') as f:
        pickle.dump(logprobs, f)

    tok_guesses, tok_confidences, tok_none_position = parse_scores(logprobs, answer_key='Guess')
    ask_guesses, ask_confidences, ask_none_position = parse_scores(generated_text)

    for j in set(ask_none_position + tok_none_position):
        ask_guesses[j] = random.choice(['Yes', 'No'])
        ask_confidences[j] = 0
        tok_guesses[j] = ask_guesses[j]
        tok_confidences[j] = ask_confidences[j]

    output_df['guess'] = ask_guesses
    output_df['ask_confidence'] = ask_confidences
    output_df['tok_confidence'] = tok_confidences

    output_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)
        random.seed(args.seed)

    logging.set_verbosity_info()

    main(args)

