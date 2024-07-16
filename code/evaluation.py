import random

import pandas as pd
import argparse
import re
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, brier_score_loss, ndcg_score, \
    accuracy_score, f1_score
from scipy.stats import kendalltau
import calibration as cal
from rank_eval import Qrels, Run, evaluate
import pickle
from inference import find_first_float

random.seed(42)


def evaluate_ranker_results():
    df = pd.read_csv('data/chatreport_test_results.csv')
    prob_large = df['large_embed_score'].tolist()
    prob_small = df['small_embed_score'].tolist()
    prob_ada = df['ada_embed_score'].tolist()
    prob_bge_large = df['large_scores'].tolist()
    prob_bge_base = df['base_scores'].tolist()
    gpt35_listwise_20 = df['gpt35_rerank_20'].tolist()
    gpt35_listwise_bg_20 = df['gpt35_rerank_bg_20'].tolist()
    gpt4_listwise_20 = df['gpt4_rerank_20'].tolist()
    gpt4_listwise_bg_20 = df['gpt4_rerank_bg_20'].tolist()
    gpt35_listwise_2 = df['gpt35_rerank_2'].tolist()
    gpt35_listwise_bg_2 = df['gpt35_rerank_bg_2'].tolist()
    gpt4_listwise_2 = df['gpt4_rerank_2'].tolist()
    gpt4_listwise_bg_2 = df['gpt4_rerank_bg_2'].tolist()
    gpt35_listwise = df['gpt35_rerank'].tolist()
    gpt35_listwise_bg = df['gpt35_rerank_bg'].tolist()
    gpt4_listwise = df['gpt4_rerank'].tolist()
    gpt4_listwise_bg = df['gpt4_rerank_bg'].tolist()
    prob_rerank = [1 / (1 + np.exp(-x)) for x in df['rerank_scores']]
    df_test = pd.read_csv('data/chatreport_test.csv')

    probs = [prob_large, prob_small, prob_ada, prob_rerank, prob_bge_large, prob_bge_base, gpt35_listwise,
             gpt35_listwise_bg, gpt4_listwise, gpt4_listwise_bg,
             gpt35_listwise_2, gpt35_listwise_bg_2, gpt4_listwise_2, gpt4_listwise_bg_2, gpt35_listwise_20,
             gpt35_listwise_bg_20, gpt4_listwise_20, gpt4_listwise_bg_20]
    settings = ['embed_large', 'embed_small', 'embed_ada', 'rerank_gemma', 'bge_large', 'bge_small', 'gpt35_rerank',
                'gpt35_rerank_bg', 'gpt4_rerank', 'gpt4_rerank_bg',
                'gpt35_rerank_2', 'gpt35_rerank_bg_2', 'gpt4_rerank_2', 'gpt4_rerank_bg_2',
                'gpt35_rerank_20', 'gpt35_rerank_bg_20', 'gpt4_rerank_20', 'gpt4_rerank_bg_20']
    for s, prob in zip(settings, probs):
        questions = df_test.Questions.tolist()
        all_questions = []
        for q in questions:
            if q in all_questions:
                continue
            else:
                all_questions.append(q)
        gold_binary = [1 if 'no' not in l.lower() else 0 for l in df_test.gold]
        if 'gpt' not in s:
            calibration_auc = roc_auc_score(np.array(gold_binary), prob)
            calibration_ece = cal.get_ece(np.array(prob), np.array(gold_binary), num_bins=10)
            calibration_brier = brier_score_loss(np.array(gold_binary), np.array(prob))
        else:
            calibration_auc = 0
            calibration_ece = 0
            calibration_brier = 0


        gold_qrels = Qrels()
        gold_bin_qrels = Qrels()
        student_runs = Run()
        all_student_scores = []
        gold_labels = []
        gold_bin_labels = []
        q_ids = []
        all_doc_ids = []
        kendall_taus_gold_student = []
        for i, q in enumerate(all_questions):
            q_ids.append('q_' + str(i))
            # labels = [1 if 'yes' in l.lower() else 0 for l in df_test.loc[df_test['Questions'] == q, 'gold_binary']]
            labels = []
            binary_labels = []
            for l in df_test.loc[df_test['Questions'] == q, 'gold']:
                if 'yes' in l.lower():
                    labels.append(1)
                    binary_labels.append(1)
                elif 'no' in l.lower():
                    labels.append(0)
                    binary_labels.append(0)
                else:
                    labels.append(0.5)
                    binary_labels.append(1)
            doc_ids = []
            for j, _ in enumerate(labels):
                doc_ids.append('d_' + str(i) + '_' + str(j))
            all_doc_ids.append(doc_ids)
            student_scores = df_test.loc[df_test['Questions'] == q, 'prob'].tolist()

            gold_labels.append(labels)
            gold_bin_labels.append(binary_labels)
            all_student_scores.append(student_scores)
            kendall_taus_gold_student.append(kendalltau(student_scores, labels, variant='b').statistic)

        kendall_tau_gold_student = sum(kendall_taus_gold_student) / len(kendall_taus_gold_student)
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
        gold_bin_qrels.add_multi(
            q_ids=q_ids,
            doc_ids=all_doc_ids,
            scores=gold_bin_labels,
        )

        student_gold = evaluate(gold_qrels, student_runs, ["ndcg", "ndcg@10"])
        student_bin_gold = evaluate(gold_bin_qrels, student_runs, ["map", "map@10"])
        print('student golden', student_gold)

        new_df = pd.DataFrame({
            "setting": s,
            "vague_ap": None,
            "vague_auc": None,
            "binary_acc": None,
            "calibration_avg": round((calibration_auc + 2 - calibration_ece - calibration_brier) / 3, 6),
            "calibration_auc": round(calibration_auc, 6),
            "calibration_ece": round(calibration_ece, 6),
            "calibration_brier": round(calibration_brier, 6),
            "rank_gold_ndcg": round(student_gold['ndcg'], 6),
            "rank_gold_map": round(student_bin_gold['map'], 6),
            "rank_gold_kentall": round(kendall_tau_gold_student, 6),
            "rank_gold_avg": round((student_bin_gold['map'] + student_gold['ndcg']) / 2, 6),
        }, index=[0])
        if os.path.isfile('results_embed.xlsx'):
            df = pd.read_excel('results_embed.xlsx')
            df = pd.concat([df, new_df])
        else:
            df = new_df

        df.to_excel('results_embed.xlsx', index=False)


def evaluate_chatgpt_results():
    df = pd.read_csv('data/chatreport_test_results.csv')
    gpt4_guess_no_cot = df['gpt4_no_cot_guess']
    gpt4_conf_no_cot = df['gpt4_no_cot_confidence']
    gpt4_conf_tok = df['gpt4_probabYesNo'].tolist()
    gpt4_guess_tok = df['gpt4_guess'].tolist()
    gpt4_conf_nb = df['gpt4_no_explanation_confidence'].tolist()
    gpt4_guess_nb = df['gpt4_no_explanation_guess'].tolist()
    gpt35_conf_tok = df['gpt3.5_probabYesNo'].tolist()
    gpt35_guess_tok = df['gpt3.5_guess'].tolist()
    df_test = pd.read_csv('data/chatreport_test.csv')
    gpt4_guess_ask = df_test['gpt4_guess'].tolist()
    gpt4_conf_ask = df_test['gpt4_confidence'].tolist()
    gpt35_guess_ask = df_test['gpt35_guess'].tolist()
    gpt35_conf_ask = df_test['gpt35_confidence'].tolist()

    settings = ['gpt4_ask', 'gpt4_tok', 'gpt4_nb', 'gpt4_no_cot', 'gpt35_ask', 'gpt35_tok']
    gs = [gpt4_guess_ask, gpt4_guess_tok, gpt4_guess_nb, gpt4_guess_no_cot, gpt35_guess_ask,
          gpt35_guess_tok]
    cs = [gpt4_conf_ask, gpt4_conf_tok, gpt4_conf_nb, gpt4_conf_no_cot, gpt35_conf_ask, gpt35_conf_tok]

    for s, g, c in zip(settings, gs, cs):
        if 'prob' in s:
            confidences = []
            for p, gu in zip(c, g):
                p = find_first_float(str(p))
                if 'no' in gu.lower():
                    confidences.append((1 - p))
                else:
                    confidences.append(p)
            guesses = g
        else:
            confidences = np.array(c).astype(float) / 100 if 'tok' in s else c
            guesses = g
        prob = []
        for co, gu in zip(confidences, guesses):
            if 'yes' in gu.lower():
                prob.append(co)
            else:
                prob.append(1 - co)
        df_test['prob'] = prob

        questions = df_test.Questions.tolist()
        all_questions = []
        for q in questions:
            if q in all_questions:
                continue
            else:
                all_questions.append(q)
        gold_binary = [1 if 'no' not in l.lower() else 0 for l in df_test.gold]
        hard = df_test.hard.tolist()
        binary_label = [1 if 'yes' in l.lower() else 0 for l in guesses]
        acc = accuracy_score(gold_binary, binary_label)
        f1 = f1_score(gold_binary, binary_label)

        hard_auc = roc_auc_score(np.array(hard), 1 - np.array(confidences))
        hard_ap = average_precision_score(np.array(hard), 1 - np.array(confidences))
        calibration_auc = roc_auc_score(np.array(gold_binary), prob)
        relevance_ap = average_precision_score(np.array(gold_binary), prob)

        calibration_ece = cal.get_ece(np.array(prob), np.array(gold_binary), num_bins=10)
        calibration_brier = brier_score_loss(np.array(gold_binary), np.array(prob))

        gold_qrels = Qrels()
        gold_bin_qrels = Qrels()
        student_runs = Run()
        all_student_scores = []
        gold_labels = []
        gold_bin_labels = []
        q_ids = []
        all_doc_ids = []

        kendall_taus_gpt4_student = []
        kendall_taus_gold_student = []
        for i, q in enumerate(all_questions):
            q_ids.append('q_' + str(i))
            # labels = [1 if 'yes' in l.lower() else 0 for l in df_test.loc[df_test['Questions'] == q, 'gold_binary']]
            labels = []
            binary_labels = []
            for l in df_test.loc[df_test['Questions'] == q, 'gold']:
                if 'yes' in l.lower():
                    labels.append(1)
                    binary_labels.append(1)
                elif 'no' in l.lower():
                    labels.append(0)
                    binary_labels.append(0)
                else:
                    labels.append(0.5)
                    binary_labels.append(1)
            doc_ids = []
            for j, _ in enumerate(labels):
                doc_ids.append('d_' + str(i) + '_' + str(j))
            all_doc_ids.append(doc_ids)
            student_scores = df_test.loc[df_test['Questions'] == q, 'prob'].tolist()
            gpt4_scores = df_test.loc[df_test['Questions'] == q, 'gpt4_prob'].tolist()

            gold_labels.append(labels)
            gold_bin_labels.append(binary_labels)
            all_student_scores.append(student_scores)
            kendall_taus_gold_student.append(kendalltau(student_scores, labels, variant='b', nan_policy='omit').statistic)

        kendall_tau_gold_student = sum(kendall_taus_gold_student) / len(kendall_taus_gold_student)
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
        gold_bin_qrels.add_multi(
            q_ids=q_ids,
            doc_ids=all_doc_ids,
            scores=gold_bin_labels,
        )

        student_gold = evaluate(gold_qrels, student_runs, ["ndcg", "ndcg@10"])
        student_bin_gold = evaluate(gold_bin_qrels, student_runs, ["map", "map@10"])

        new_df = pd.DataFrame({
            "setting": s,
            "vague_ap": round(hard_ap, 6),
            "vague_auc": round(hard_auc, 6),
            "binary_acc": round(acc, 6),
            "binary_f1": round(f1, 6),
            "relevance_ap": round(relevance_ap, 6),
            "calibration_avg": round((calibration_auc + 2 - calibration_ece - calibration_brier) / 3, 6),
            "calibration_auc": round(calibration_auc, 6),
            "calibration_ece": round(calibration_ece, 6),
            "calibration_brier": round(calibration_brier, 6),
            "rank_gold_ndcg": round(student_gold['ndcg'], 6),
            "rank_gold_map": round(student_bin_gold['map'], 6),
            "rank_gold_kentall": round(kendall_tau_gold_student, 6),
            "rank_gold_avg": round((student_bin_gold['map'] + student_gold['ndcg']) / 2, 6),
        }, index=[0])
        if os.path.isfile('results_chatgpt.xlsx'):
            df = pd.read_excel('results_chatgpt.xlsx')
            df = pd.concat([df, new_df])
        else:
            df = new_df

        df.to_excel('results_chatgpt.xlsx', index=False)


if __name__ == '__main__':
    evaluate_ranker_results()
    evaluate_chatgpt_results()
