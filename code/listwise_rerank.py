import os
import openai
from openai import AsyncOpenAI
import argparse
import time
import asyncio

import random
import re
import pandas as pd
from dotenv import load_dotenv
from get_training_data import MODEL_DICT, create_answers_async, get_input_price, get_output_price


SYSTEM = """You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query."""

USER = """I will provide you with {num} passages, each indicated by a numerical identifier [].
Rank the passages based on their relevance to the search query: {query}.
{passages}
Search Query: {query}.
Rank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.
"""

USER_BG = """I will provide you with {num} passages, each indicated by a numerical identifier [].
Rank the passages based on their relevance to the search query: {query}.
{passages}
Search Query: {query}.

Here are some background information that explains the query: {background}

Rank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain.
"""

load_dotenv(dotenv_path='apikey.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)


def get_rank_from_response(res, rank_end, rank_start):
    rand_rank = list(range(1, rank_end - rank_start + 2))[::-1]
    splits = res.strip().split('>')
    if len(splits) != rank_end - rank_start + 1:
        return rand_rank, True
    ret_rank = []
    for num in splits:
        match = re.search(r'\[(\d+)\]', num)
        if match:
            ret_rank.append(int(match.group(1)) + 1)
        else:
            return rand_rank, True
    for j in rand_rank:
        if j not in ret_rank:
            return rand_rank, True
    return ret_rank, False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--num_doc_per_question", type=int, default=60)
    args = parser.parse_args()

    # Listwise rerank requires first sorting documents with embedding models, here we use openai text-embedding-3-small
    df = pd.read_csv('chatreport_test_sorted.csv')
    all_questions = []
    all_background = []
    for q, b in zip(df['Questions'], df['Background']):
        if q not in all_questions:
            all_questions.append(q)
            all_background.append(b)
    print(len(all_questions))

    if 'relevance' not in df.columns:
        df['relevance'] = 0
        for q in all_questions:
            df.loc[df['Questions'] == q, 'relevance'] = list(range(60))

    rank_start_list = list(range(0, args.num_doc_per_question - args.step_size, args.step_size))
    rank_end_list = [i + 2 * args.step_size - 1 for i in rank_start_list]

    all_prompts = []
    all_responses = []
    async_client = AsyncOpenAI()

    for id, (rank_start, rank_end) in enumerate(zip(rank_start_list, rank_start_list)):
        prompts = []
        input_prices = []
        user_messages = []
        for q, b in zip(all_questions, all_background):
            paragraphs = df.loc[
                (df['relevance'] >= rank_start) & (df['relevance'] <= rank_end) & (df['Questions'] == q) & (
                            df['Questions'] == q), 'Paragraphs'].tolist()
            para_string = ""
            for i, p in enumerate(paragraphs):
                para_string += "[{0}] {1}\n".format(i, p)
            message = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_BG.format(passages=para_string, query=q, num=int(rank_end - rank_start + 1),
                                                   background=b)},
            ]
            prompts.append(message)
            user_messages.append(
                USER_BG.format(passages=para_string, query=q, num=int(rank_end - rank_start + 1), background=b))
            input_prices.append(get_input_price(
                SYSTEM + USER_BG.format(passages=para_string, query=q, num=int(rank_end - rank_start + 1),
                                        background=b), MODEL_DICT['gpt4']))
        all_prompts.extend(user_messages)

        print(sum(input_prices))


        def batchify(lst, batch_size):
            """Split the list `lst` into sublists of size `batch_size`."""
            return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


        batched_prompts = batchify(prompts, batch_size=6)

        final_answers = []
        start_batch = 0
        for i, b in enumerate(batched_prompts):
            if i < start_batch:
                continue
            answers = asyncio.run(create_answers_async(async_client, model=MODEL_DICT['gpt4'], messages=b))
            final_answers.extend(answers)
            time.sleep(1)

        generated_text = []
        for output in final_answers:
            generated_text.append(output.message.content)
        all_responses.extend(generated_text)
        print(generated_text[0])
        output_df = pd.DataFrame({'prompt': prompts, 'output': generated_text})
        output_df.to_csv('gpt4_rerank_' + str(id) + '.csv', index=False)
        error_count = 0
        for q, response in zip(all_questions, generated_text):
            result_rank, error = get_rank_from_response(response, rank_end, rank_start)
            if error:
                error_count += 1
                new_relevance = df.loc[(df['relevance'] >= rank_start) & (df['relevance'] <= rank_end) & (
                            df['Questions'] == q), 'relevance'].tolist()
            else:
                new_relevance = [rank_end - result_rank.index(i) for i in range(1, rank_end - rank_start + 2)]
            # print(df.loc[(df['relevance'] >= rank_start) & (df['relevance'] <= rank_end) & (df['Questions'] == q)])
            print(new_relevance)
            df.loc[(df['relevance'] >= rank_start) & (df['relevance'] <= rank_end) & (
                        df['Questions'] == q), 'relevance'] = new_relevance
        print(df['relevance'].tolist())

    df_results = pd.DataFrame({'prompts': all_prompts, 'responses': all_responses})
    df_results.to_csv('gpt4_listwise_bg.csv')


