import os
import time
import sys

import openai
from openai import OpenAI, AsyncOpenAI
import asyncio
import random
import pandas as pd
import tiktoken
from dotenv import load_dotenv
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

load_dotenv(dotenv_path='apikey.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

# Following commented out prompts are used in paper (https://arxiv.org/abs/2406.14162) to annotate ChatReport data.
# SYSTEM = "You are a helpful assistant who assists human analysts in identifying useful information within sustainability reports for their analysis."
SYSTEM = "You are a helpful assistant who assists human analysts in identifying useful information for their analysis."
USER_PROMPT = """You will be provided with a <question> the analyst seeks to answer, a <paragraph> extracted from a lengthy report, and <background_information> that explains the <question>. <background_information> first explains the <question> and then raises examples to help you to better understand the <question>. Your job is to assess whether the <paragraph> is useful in answering the <question>.

<background_information>: "{background_information}"
<question>: "{question}"
<paragraph>: "{paragraph_chunk}"


Is <paragraph> helpful for answering <question>? Note that the <paragraph> can be helpful even it only addresses part of the <question> without fully answering it. Provide your best guess for this question and your confidence that the guess is correct. Reply in the following format:
[Reason]: <Reason why and how the paragraph is helpful or not helpful for answering the question. Clearly indicate your stance.>
[Guess]: <Your most likely guess, should be one of "Yes" or "No".>
[Confidence]: <Give your honest confidence score between 0.0 and 1.0 about the correctness of your guess. 0 means your previous guess is very likely to be wrong, and 1 means you are very confident about the guess.>"""

TRAIN_PROMPT = """You will be provided with a <question> the analyst seeks to answer, a <paragraph> extracted from a lengthy report, and <background_information> that explains the <question>. <background_information> first explains the <question> and then raises examples to help you to better understand the <question>. Your job is to assess whether the <paragraph> is useful in answering the <question>.

<background_information>: "{background_information}"
<question>: "{question}"
<paragraph>: "{paragraph_chunk}"


Is <paragraph> helpful for answering <question>? Note that the <paragraph> can be helpful even it only addresses part of the <question> without fully answering it. Provide your best guess for this question and your confidence that the guess is correct. Reply in the following format:
[Guess]: <Your most likely guess whether the paragraph is helpful or not helpful for answering the question, should be one of "Yes" or "No".>
[Confidence]: <Give your honest confidence score between 0.0 and 1.0 about the correctness of your guess. 0 means your previous guess is very likely to be wrong, and 1 means you are very confident about the guess.>"""

TRAIN_RESPONSE = """[Guess]: {guess}

[Confidence]: {confidence}"""


def format_instruction(tokenizer, system, input, output, no_system=False):
    if no_system:
        chat_full = [
            {"role": "user", "content": system + '\n\n' +input},
            {"role": "assistant", "content": output},
        ]
        chat_prompt = [
            {"role": "user", "content": system + '\n\n' +input},
        ]
    else:
        chat_full = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
            {"role": "assistant", "content": output},
        ]
        chat_prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    formatted_input = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    formatted_full = tokenizer.apply_chat_template(chat_full, tokenize=False, add_generation_prompt=False)
    formatted_output = formatted_full.replace(formatted_input, '')
    return formatted_input, formatted_output


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def achat(client, model, message):
    out = await client.chat.completions.create(
        messages=message,
        model=model,
        seed=42,
        temperature=0,
    )
    return out.choices[0].message.content


async def create_answers_async(client, model, messages, batch_size=20):
    # async answering
    batched_msgs = batchify(messages, batch_size)
    all_answers = []
    for i, batch in enumerate(batched_msgs):
        answers = await asyncio.gather(*[achat(client, model, m) for m in batch])
        all_answers.extend(answers)
        print(f"Batch {i} Answers Given")
        time.sleep(1)
    return all_answers


def get_input_price(input_str, model):
    encoding = tiktoken.encoding_for_model(MODEL_DICT[model])
    input_len = len(encoding.encode(input_str))
    input_cost = input_len / 1000000 * INPUT_COST_DICT[model]
    return input_cost


def get_output_price(output_str, model):
    encoding = tiktoken.encoding_for_model(MODEL_DICT[model])
    output_len = len(encoding.encode(output_str))
    output_cost = output_len / 1000000 * OUTPUT_COST_DICT[model]
    return output_cost


async def get_embeddings(client, texts, model='text-embedding-3-small', batch_size=1000):
    batched_texts = batchify(texts, batch_size)
    embeddings = []
    for batch in batched_texts:
        batch_embeddings = await client.embeddings.create(input=batch, model=model)
        embeddings.extend([e.embedding for e in batch_embeddings.data])
        # Sleep to avoid API rate limitation, may remove this line or increase the batch_size to speed up.
        time.sleep(1)
    return np.array(embeddings)


def top_related_documents(query_embeddings, document_embeddings, top_n=5):
    # Calculate cosine similarity between query embeddings and document embeddings
    similarities = cosine_similarity(query_embeddings, document_embeddings)

    # Get the indices of the top-n related documents for each query
    top_related_indices = [(-similarities[i]).argsort()[:top_n].tolist() for i in range(similarities.shape[0])]

    return top_related_indices


MODEL_DICT = {
    'gpt35': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4o',
}
INPUT_COST_DICT = {
    'gpt35': 0.5,
    'gpt4': 5,
}
OUTPUT_COST_DICT = {
    'gpt35': 1.5,
    'gpt4': 15,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, default="example/question.jsonl")
    parser.add_argument("--document_file", type=str, default="example/document.jsonl")
    parser.add_argument("--output_file", type=str, default="example/distilled_train_data.xlsx")
    parser.add_argument("--teacher_llm", type=str, default="gpt4", choices=['gpt35', 'gpt4'])
    parser.add_argument("--sample_num", type=int, default=30)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--student_llm", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--cache", type=str, default=".")
    parser.add_argument("--huggingface_token", type=str, default="")
    args = parser.parse_args()

    df = pd.read_json(args.question_file, lines=True)
    df_docs = pd.read_json(args.document_file, lines=True)
    try:
        assert ('explanation' in df.columns and 'question' in df.columns)
    except AssertionError as e:
        print("Either explanation or question is not available in the question file! Please first run "
              "get_question_explanation to generate relevance definition.")
        sys.exit(1)
    assert ('document' in df_docs.columns and 'report' in df_docs.columns)
    questions = df['question'].tolist()
    async_client = AsyncOpenAI()

    # Balanced sampling in and out of Top-5 of all reports.
    unique_reports = []
    for r in df_docs['report']:
        if r not in unique_reports:
            unique_reports.append(r)
    all_question_top = {}
    all_question_non_top = {}
    for report in unique_reports:
        documents = df_docs.loc[df_docs['report'] == report, 'document'].tolist()
        doc_embeddings = asyncio.run(get_embeddings(async_client, documents, model=MODEL_DICT[args.teacher_llm]))
        question_embeddings = asyncio.run(get_embeddings(async_client, questions, model=MODEL_DICT[args.teacher_llm]))
        top_document_ids = top_related_documents(question_embeddings, doc_embeddings, top_n=args.top_n)
        for i, q in enumerate(questions):
            top_for_q_i = [d for i, d in enumerate(documents) if i in top_document_ids[i]]
            non_top_for_q_i = [d for i, d in enumerate(documents) if i not in top_document_ids[i]]
            if q not in all_question_top.keys():
                all_question_top[q] = top_for_q_i
                all_question_non_top[q] = non_top_for_q_i
            else:
                all_question_top[q].extend(top_for_q_i)
                all_question_non_top[q].extend(non_top_for_q_i)
    train_data_questions = []
    train_data_documents = []

    for k, v in all_question_top.items():
        sampled_docs = random.sample(v, args.sample_num)
        for d in sampled_docs:
            train_data_questions.append(k)
            train_data_documents.append(d)

    for k, v in all_question_non_top.items():
        sampled_docs = random.sample(v, args.sample_num)
        for d in sampled_docs:
            train_data_questions.append(k)
            train_data_documents.append(d)

    # After sampling, we can distill training data from GPT-4
    prompts = [USER_PROMPT.format(background_information=df.loc[df['question'] == q, 'explanation'], question=q,
                                  paragraph_chunk=p) for q, p in zip(train_data_questions, train_data_documents)]
    parsed_prompts = []
    input_prices = []
    for p in prompts:
        message = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p},
        ]
        parsed_prompts.append(message)
        input_prices.append(get_input_price(SYSTEM + p, args.teacher_llm))

    print(sum(input_prices))

    responses = asyncio.run(create_answers_async(async_client, model=MODEL_DICT[args.teacher_llm], messages=parsed_prompts))
    reasons = []
    guesses = []
    confidence_scores = []
    output_costs = []
    for i, response in enumerate(responses):
        output_costs.append(get_output_price(response, args.teacher_llm))
        content = response.strip('\n').split('\n')
        reason = None
        guess = None
        confidence = None
        for c in content:
            if '[Reason]:' in c:
                reason = c.split('[Reason]:')[-1].strip()
            if '[Guess]:' in c:
                guess = c.split('[Guess]:')[-1].strip()
            if '[Confidence]:' in c:
                confidence = c.split('[Confidence]:')[-1].strip()
        if reason is None or guess is None or confidence is None:
            print("Format error in index ", i)
            print(response)
        reasons.append(str(reason))
        guesses.append(str(guess))
        confidence_scores.append(str(confidence))

    print("Output cost: ", output_costs)
    pd.DataFrame({'answers': responses, 'reason': reasons, 'guess': guesses, 'confidence': confidence_scores}).to_excel(args.output_file)
    train_prompts = [TRAIN_PROMPT.format(background_information=df.loc[df['question'] == q, 'explanation'], question=q,
                     paragraph_chunk=p) for q, p in zip(train_data_questions, train_data_documents)]
    train_responses = [TRAIN_RESPONSE.format(guess=g, confidence=c) for g, c in zip(guesses, confidence_scores)]

    tokenizer = AutoTokenizer.from_pretrained(args.student_llm, cache_dir=args.cache, token=args.huggingface_token)
    no_system = False
    if "gemma" in args.student_llm:
        no_system = True

    formatted_prompts = []
    formatted_responses = []
    for train_prompt, train_response in zip(train_prompts, train_responses):
        formatted_prompt, formatted_response = format_instruction(tokenizer, SYSTEM, train_prompt, train_response, no_system=no_system)
        formatted_prompts.append(formatted_prompt)
        formatted_responses.append(formatted_response)

    pd.DataFrame({'input': formatted_prompts, 'output': formatted_responses}).to_csv(args.output_file.replace('.xlsx', '.csv'), index=False)