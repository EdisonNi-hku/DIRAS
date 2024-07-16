import random

import pandas as pd
import argparse
import re
import os
import numpy as np
from rank_eval import Qrels, Run, evaluate
import pickle
from scipy.stats import kendalltau
from inference import parse_texts, parse_scores


def main():
    df = pd.read_excel('data/climretrieve_relevant.xlsx')
    gpt4_generic_guesses = df.gpt4_generic_guess.tolist()
    gpt4_generic_confidences = df.gpt4_generic_confidence.tolist()
    gpt35_generic_guesses = df.gpt35_generic_guess.tolist()
    gpt35_generic_confidences = df.gpt35_generic_confidence.tolist()
    llama_generic_guesses = df.llama_generic_guess.tolist()
    llama_generic_ask = df.llama_generic_ask_confidence.tolist()
    llama_generic_tok = df.llama_generic_tok_confidence.tolist()
    gpt4_specific_guesses = df.gpt4_specific_guess.tolist()
    gpt4_specific_confidences = df.gpt4_specific_confidence.tolist()
    gpt35_specific_guesses = df.gpt35_specific_guess.tolist()
    gpt35_specific_confidences = df.gpt35_specific_confidence.tolist()
    llama_specific_guesses = df.llama_specific_guess.tolist()
    llama_specific_ask = df.llama_specific_ask_confidence.tolist()
    llama_specific_tok = df.llama_specific_tok_confidence.tolist()
    small_embed = df.small_embed.tolist()
    large_embed = df.large_embed.tolist()
    random.seed(40)
    random_40 = [random.random() for _ in range(len(llama_specific_tok))]
    random.seed(41)
    random_41 = [random.random() for _ in range(len(llama_specific_tok))]
    random.seed(42)
    random_42 = [random.random() for _ in range(len(llama_specific_tok))]
    random.seed(43)
    random_43 = [random.random() for _ in range(len(llama_specific_tok))]
    random.seed(44)
    random_44 = [random.random() for _ in range(len(llama_specific_tok))]
    random.seed(42)

    settings = ['gpt4_generic', 'gpt4_specific', 'gpt35_generic', 'gpt35_specific', 'llama_generic_ask',
                'llama_generic_tok', 'llama_specific_ask', 'llama_specific_tok', 'small_embed', 'large_embed',
                'random_40', 'random_41','random_42', 'random_43', 'random_44']
    all_guesses = [gpt4_generic_guesses, gpt4_specific_guesses, gpt35_generic_guesses, gpt35_specific_guesses,
                   llama_generic_guesses, llama_generic_guesses, llama_specific_guesses, llama_specific_guesses, None, None,
                   None, None, None, None, None]
    all_confidences = [gpt4_generic_confidences, gpt4_specific_confidences, gpt35_generic_confidences, gpt35_specific_confidences,
                       llama_generic_ask, llama_generic_tok, llama_specific_ask, llama_specific_tok, small_embed, large_embed,
                       random_40, random_41, random_42, random_43, random_44]

    for setting, guesses, confidences in zip(settings, all_guesses, all_confidences):
        if setting == 'random':
            prob = confidences
        elif guesses is None:
            prob = confidences
        else:
            prob = []
            for c, g in zip(confidences, guesses):
                if 'yes' in g.lower():
                    prob.append(c)
                else:
                    prob.append(1 - c)
        df_test = pd.read_excel('data/climretrieve_relevant.xlsx')
        df_test['prob'] = prob
        df_test['question'] = df_test['question'].apply(lambda x: x.lower().strip())
        questions = df_test['question'].tolist()
        all_questions = []
        for q in questions:
            if q in all_questions:
                continue
            else:
                all_questions.append(q)

        df_test['gold'] = [int(s) / 3 for s in df_test['relevance']]
        # df_test['gold'] = [0 if int(s) < 2 else 1 for s in df_test['relevance']]

        gold_qrels = Qrels()
        student_runs = Run()
        all_student_scores = []
        gold_labels = []
        q_ids = []
        all_doc_ids = []
        for i, q in enumerate(all_questions):
            q_ids.append('q_' + str(i))
            labels = df_test.loc[df_test['question'] == q, 'gold'].tolist()
            doc_ids = []
            for j, _ in enumerate(labels):
                doc_ids.append('d_' + str(i) + '_' + str(j))
            all_doc_ids.append(doc_ids)
            student_scores = df_test.loc[df_test['question'] == q, 'prob'].tolist()

            gold_labels.append(labels)
            all_student_scores.append(student_scores)

        gold_qrels.add_multi(
            q_ids=q_ids,
            doc_ids=all_doc_ids,
            scores=gold_labels,
        )
        student_runs.add_multi(
            q_ids=q_ids,
            doc_ids=all_doc_ids,
            scores=all_student_scores,
        )

        metrics = ["ndcg", "ndcg@5", "ndcg@10", "ndcg@15"]
        student_gold = evaluate(gold_qrels, student_runs, metrics)
        print('student golden', student_gold)

        output_name = 'results_ndcg_only_gold.xlsx'
        new_df = pd.DataFrame({
            "setting": setting,
            "rank_gold_ndcg": round(student_gold['ndcg'], 6),
            "rank_gold_ndcg@5": round(student_gold['ndcg@5'], 6),
            "rank_gold_ndcg@10": round(student_gold['ndcg@10'], 6),
            "rank_gold_ndcg@15": round(student_gold['ndcg@15'], 6),
        }, index=[0])
        if os.path.isfile(output_name):
            df = pd.read_excel(output_name)
            df = pd.concat([df, new_df])
        else:
            df = new_df

        df.to_excel(output_name, index=False)


if __name__ == '__main__':
    main()