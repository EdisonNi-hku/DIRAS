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
from evaluation import find_first_float

random.seed(42)


def parse_texts(outputs, splitter):
    none_count = 0
    guesses = []
    confidences = []
    none_position = []
    for i, output in enumerate(outputs):
        response = output.split(splitter, 1)[-1]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    if 'gemma' in args.file:
        splitter = "<start_of_turn>model\n"
    elif 'phi3' in args.file:
        splitter = "<|assistant|>"
    else:
        splitter = "<|start_header_id|>assistant<|end_header_id|>\n"

    if 'rgc' in args.file or 'gc' in args.file:
        answer_key = 'Guess'
    else:
        answer_key = 'Answer'
    outputs = pd.read_csv(args.file.split('.')[0] + '.csv')['output'].tolist()
    ask_guesses, ask_confidences, ask_none_position = parse_texts(outputs, splitter=splitter)

    with open(args.file.split('.')[0] + '.pkl', 'rb') as f:
        outputs = pickle.load(f)
    tok_guesses, tok_confidences, tok_none_position = parse_scores(outputs, answer_key=answer_key)

    if 'rgc' in args.setting_name or 'gc' in args.setting_name:
        for j in set(ask_none_position + tok_none_position):
            ask_guesses[j] = random.choice(['Yes', 'No'])
            ask_confidences[j] = 0
            tok_guesses[j] = ask_guesses[j]
            tok_confidences[j] = ask_confidences[j]

    if 'tok' in args.setting_name:
        confidences = tok_confidences
        guesses = tok_guesses
    else:
        confidences = ask_confidences
        guesses = ask_guesses

    binary_label = []
    prob = []
    for c, g in zip(confidences, guesses):
        if 'yes' in g.lower():
            prob.append(c)
            binary_label.append(1)
        else:
            prob.append(1 - c)
            binary_label.append(0)
    df_test = pd.read_csv('data/chatreport_test_results.csv')
    df_test['prob'] = prob
    questions = df_test.Questions.tolist()
    all_questions = []
    for q in questions:
        if q in all_questions:
            continue
        else:
            all_questions.append(q)
    gpt4_probs = df_test.gpt4_prob.tolist()
    gold_binary = df_test.gold_binary.tolist()
    gold_binary = [1 if 'yes' in l.lower() else 0 for l in gold_binary]
    correct_binary = []
    for gold, bi in zip(gold_binary, binary_label):
        if gold == bi:
            correct_binary.append(1)
        else:
            correct_binary.append(0)

    hard = df_test.hard.tolist()
    acc = accuracy_score(gold_binary, binary_label)
    f1 = f1_score(gold_binary, binary_label)

    hard_auc = roc_auc_score(np.array(hard), 1 - np.array(confidences))
    hard_ap = average_precision_score(np.array(hard), 1 - np.array(confidences))
    calibration_auc = roc_auc_score(np.array(correct_binary), confidences)
    relevance_ap = average_precision_score(np.array(gold_binary), prob)

    calibration_ece = cal.get_ece(np.array(prob), np.array(gold_binary), num_bins=10)
    calibration_brier = brier_score_loss(np.array(gold_binary), np.array(prob))

    gpt4_rmse = mean_squared_error(np.array(gpt4_probs), np.array(prob), squared=False)

    print('Hard auc', hard_auc)
    print('Hard ap', hard_ap)

    print('Relevance ap', relevance_ap)

    print('Calibration auc', calibration_auc)
    print('Calibration ece', calibration_ece)
    print('Calibration brier', calibration_brier)

    print('gpt4 rmse', gpt4_rmse)

    gold_qrels = Qrels()
    gold_bin_qrels = Qrels()
    gpt4_qrels = Qrels()
    gpt4_runs = Run()
    all_gpt4_scores = []
    student_runs = Run()
    all_student_scores = []
    gold_labels = []
    gold_bin_labels = []
    q_ids = []
    all_doc_ids = []
    # gpt4_gold_ndcg_scores = []
    # student_gold_ndcg_scores = []
    # student_gpt4_ndcg_scores = []
    kendall_taus_gpt4_student = []
    kendall_taus_gold_student = []
    for i, q in enumerate(all_questions):
        q_ids.append('q_' + str(i))
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
        gpt4_scores = df_test.loc[df_test['Questions'] == q, 'gpt4_prob'].tolist()
        student_scores = df_test.loc[df_test['Questions'] == q, 'prob'].tolist()

        # gpt4_gold_ndcg_scores.append(ndcg_score(labels, gpt4_scores))
        # student_gold_ndcg_scores.append(ndcg_score(labels, student_scores))
        # student_gpt4_ndcg_scores.append(ndcg_score(gpt4_scores, student_scores))

        gold_labels.append(labels)
        gold_bin_labels.append(binary_labels)
        all_gpt4_scores.append(gpt4_scores)
        all_student_scores.append(student_scores)
        kendall_taus_gold_student.append(kendalltau(student_scores, labels, variant='b').statistic)
        kendall_taus_gpt4_student.append(kendalltau(student_scores, gpt4_scores, variant='b').statistic)

    kendall_tau_gpt4_student = sum(kendall_taus_gpt4_student) / len(kendall_taus_gpt4_student)
    kendall_tau_gold_student = sum(kendall_taus_gold_student) / len(kendall_taus_gold_student)
    # print('gpt4 gold', ndcg_score(gold_labels, all_gpt4_scores))
    # student_gold_ndcg = ndcg_score(gold_labels, all_student_scores)
    # student_gpt4_ndcg = ndcg_score(all_gpt4_scores, all_student_scores)
    # print('student gold', student_gold_ndcg)
    # print('student gpt4', student_gpt4_ndcg)
    # print('student gold', np.mean(student_gold_ndcg_scores))
    # print('student gpt4', np.mean(student_gpt4_ndcg_scores))
    gold_qrels.add_multi(
        q_ids=q_ids,
        doc_ids=all_doc_ids,
        scores=gold_labels,
    )
    gold_bin_qrels.add_multi(
        q_ids=q_ids,
        doc_ids=all_doc_ids,
        scores=gold_bin_labels,
    )
    gpt4_qrels.add_multi(
        q_ids=q_ids,
        doc_ids=all_doc_ids,
        scores=all_gpt4_scores,
    )
    gpt4_runs.add_multi(
        q_ids=q_ids,
        doc_ids=all_doc_ids,
        scores=all_gpt4_scores,
    )
    student_runs.add_multi(
        q_ids=q_ids,
        doc_ids=all_doc_ids,
        scores=all_student_scores,
    )

    print('gpt4 golden', evaluate(gold_qrels, gpt4_runs, ["ndcg", "ndcg@10"]))
    student_gold = evaluate(gold_qrels, student_runs, ["ndcg", "ndcg@10"])
    student_bin_gold = evaluate(gold_bin_qrels, student_runs, ["map", "map@10"])
    student_gpt4 = evaluate(gpt4_qrels, student_runs, ["ndcg", "ndcg@10"])
    print('student golden', student_gold)
    print('student gpt4', student_gpt4)

    assert args.output.endswith('xlsx')
    new_df = pd.DataFrame({
        "setting": args.setting_name,
        "vague_ap": round(hard_ap, 6),
        "vague_auc": round(hard_auc, 6),
        "binary_acc": round(acc, 6),
        "binary_f1": round(f1, 6),
        "calibration_auc": round(calibration_auc, 6),
        "calibration_ece": round(calibration_ece, 6),
        "calibration_brier": round(calibration_brier, 6),
        "rank_gpt4_rmse": round(gpt4_rmse, 6),
        "rank_gpt4_kentall": round(kendall_tau_gpt4_student, 6),
        "rank_gpt4_ndcg": round(student_gpt4['ndcg'], 6),
        "rank_gold_ndcg": round(student_gold['ndcg'], 6),
        "rank_gold_map": round(student_bin_gold['map'], 6),
        "rank_gold_kentall": round(kendall_tau_gold_student, 6),
        "rank_gold_ndcg@10": round(student_gold['ndcg@10'], 6),
        "relevance_ap": round(relevance_ap, 6),
        "calibration_avg": round((calibration_auc + 2 - calibration_ece - calibration_brier) / 3, 6),
        "rank_gold_avg": round((student_bin_gold['map'] + student_gold['ndcg']) / 2, 6),
        "avg": round((hard_ap + f1 + (calibration_auc + 2 - calibration_ece - calibration_brier) / 3 + (student_bin_gold['map'] + student_gold['ndcg']) / 2) / 4, 6)
    }, index=[0])
    if os.path.isfile(args.output):
        df = pd.read_excel(args.output)
        df = pd.concat([df, new_df])
    else:
        df = new_df

    df.to_excel(args.output, index=False)


if __name__ == '__main__':
    main()
