import pandas as pd
import os
import openai
import random
import argparse
import asyncio

from openai import AsyncOpenAI
from dotenv import load_dotenv
from get_training_data import MODEL_DICT, create_answers_async

load_dotenv(dotenv_path='apikey.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
random.seed(42)

# GET_EXPLANATION = """An analyst posts a <question> about a sustainability report. Your task is to explain the <question> in the context of sustainability reporting. Please first explain the meaning of the <question>, i.e., meaning of the question itself and the concepts mentioned. And then give a list of examples, showing what information from the sustainability report the analyst is looking for by posting this <question>.
#
# For <the question's meaning>, please start by repeating the question in the following format:
# '''
# The question "<question>" is asking for information about [...]
# '''
#
# For the <list of example information that the question is looking for>, following the following example in terms of format:
# ---
# [...]
# 3. Initiatives aimed at creating new job opportunities in the green economy within the company or in the broader community.
# 4. Policies or practices in place to ensure that the transition to sustainability is inclusive, considering gender, race, and economic status.
# [...]
# ---
#
# Here is the question:
# <question>: ""{question}""
#
# Additionally, here is a <list of question-relevant example information> that an expert human labeler annotated. Please keep these examples in mind when answering:
# --- [BEGIN <list of question-relevant example information>]
# {examples}
# --- [END <list of question-relevant example information>]
#
# Format your reply in the following template and keep your answer concise:
#
# Meaning of the question: <the question's meaning>
# Examples of information that the question is looking for: <list of example information that the question is looking for>"""
SYSTEM = """You are a helpful AI assistant."""
GET_EXPLANATION = """An analyst posts a <question> about some report. Your task is to explain the <question>. Please first explain the meaning of the <question>, i.e., meaning of the question itself and the concepts mentioned. And then give a list of examples, showing what information from the report the analyst is looking for by posting this <question>.

For <the question's meaning>, please start by repeating the question in the following format:
'''
The question "<question>" is asking for information about [...]
'''

For the <list of example information that the question is looking for>, following the following example in terms of format:
---
[...]
3. Initiatives aimed at creating new job opportunities in the green economy within the company or in the broader community.
4. Policies or practices in place to ensure that the transition to sustainability is inclusive, considering gender, race, and economic status.
[...]
---

Here is the question:
<question>: ""{question}""

Additionally, here is a <list of question-relevant example information> that an expert human labeler annotated. Please keep these examples in mind when answering:
--- [BEGIN <list of question-relevant example information>]
{examples}
--- [END <list of question-relevant example information>]

Format your reply in the following template and keep your answer concise:

Meaning of the question: <the question's meaning>
Examples of information that the question is looking for: <list of example information that the question is looking for>"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", type=str, default="example/question.jsonl")
    args = parser.parse_args()

    df = pd.read_json(args.question_file, lines=True)
    try:
        assert ('explanation' not in df.columns)
    except AssertionError as e:
        print("There are already question explanations existing.")
    messages = []
    for q in df.question:
        message = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": GET_EXPLANATION.format(question=q)},
        ]
        messages.append(message)
    async_client = AsyncOpenAI()
    responses = asyncio.run(
        create_answers_async(async_client, model=MODEL_DICT['gpt4'], messages=messages))
    df['explanations'] = responses

    df.to_json(args.question_file, index=False, lines=True)






