import pandas as pd
import os
from rank_eval import Qrels, Run, evaluate
from scipy.stats import kendalltau
import numpy as np

file_names = ['climretrieve_all_results/llama_gc_2e_specific.csv', 'climretrieve_all_results/llama_gc_2e_generic.csv']
setting_names = ['llama_gc_specific', 'llama_gc_generic']


def table4():
    df_test = pd.read_csv('data/climretrieve_all.csv')
    questions = df_test['question'].tolist()
    reports = df_test['report'].tolist()
    all_questions = []
    for q in questions:
        if q in all_questions:
            continue
        else:
            all_questions.append(q)
    all_reports = []
    for q in reports:
        if q in all_reports:
            continue
        else:
            all_reports.append(q)
    for setting, file in zip(setting_names, file_names):

        df_result = pd.read_csv(file)
        model = 'llama_rgc' if 'rgc' in setting else 'llama_gc'
        guesses = df_result[model + '_guess']
        for conf in ['_tok', '_ask']:
            save_setting = setting + conf
            confidences = df_result[model + conf + '_confidence']
            prob = []
            for c, g in zip(confidences, guesses):
                if 'yes' in g.lower():
                    prob.append(c)
                else:
                    prob.append(1 - c)
            df_test['prob'] = prob

            df_test['gold'] = [int(str(s)[0]) / 3 for s in df_test['relevance']]
            df_test['gold_binary'] = [1 if s > 0 else 0 for s in df_test['relevance']]

            gold_qrels = Qrels()
            gold_bi_qrels = Qrels()
            student_runs = Run()
            all_student_scores = []
            gold_labels = []
            gold_labels_bi = []
            q_ids = []
            all_doc_ids = []
            for i, q in enumerate(all_questions):
                for j, r in enumerate(all_reports):
                    labels = df_test.loc[(df_test['question'] == q) & (df_test['report'] == r), 'gold'].tolist()
                    if len(labels) == 0:
                        continue
                    q_ids.append('r_' + str(j) + '_q_' + str(i))
                    labels_bi = df_test.loc[(df_test['question'] == q) & (df_test['report'] == r), 'gold_binary'].tolist()
                    doc_ids = []
                    for k, _ in enumerate(labels):
                        doc_ids.append('d_' + str(i) + '_' + str(j) + '_' + str(k))

                    all_doc_ids.append(doc_ids)
                    student_scores = df_test.loc[(df_test['question'] == q) & (df_test['report'] == r), 'prob'].tolist()

                    gold_labels.append(labels)
                    gold_labels_bi.append(labels_bi)
                    all_student_scores.append(student_scores)

            gold_qrels.add_multi(
                q_ids=q_ids,
                doc_ids=all_doc_ids,
                scores=gold_labels,
            )
            gold_bi_qrels.add_multi(
                q_ids=q_ids,
                doc_ids=all_doc_ids,
                scores=gold_labels_bi,
            )
            student_runs.add_multi(
                q_ids=q_ids,
                doc_ids=all_doc_ids,
                scores=all_student_scores,
            )

            student_gold = evaluate(gold_qrels, student_runs, ["ndcg", "ndcg@5", "ndcg@10", "ndcg@15"])
            student_gold_bi = evaluate(gold_bi_qrels, student_runs, ["map", "map@5", "map@10", "map@15"])
            print('student golden', student_gold)

            new_df = pd.DataFrame({
                "setting": save_setting,
                "ndcg": round(student_gold['ndcg'], 6),
                "ndcg@5": round(student_gold['ndcg@5'], 6),
                "ndcg@10": round(student_gold['ndcg@10'], 6),
                "ndcg@15": round(student_gold['ndcg@15'], 6),
                "map": round(student_gold_bi['map'], 6),
                "map@5": round(student_gold_bi['map@5'], 6),
                "map@10": round(student_gold_bi['map@10'], 6),
                "map@15": round(student_gold_bi['map@15'], 6),
            }, index=[0])
            if os.path.isfile('table4.xlsx'):
                df = pd.read_excel('table4.xlsx')
                df = pd.concat([df, new_df])
            else:
                df = new_df

            df.to_excel('table4.xlsx', index=False)


def table5():
    df_test = pd.read_csv('data/climretrieve_all.csv')
    df_specific = pd.read_csv('climretrieve_all_results/llama_gc_2e_specific.csv')
    specific_guesses = df_specific['llama_gc_guess'].tolist()
    specific_confidences = df_specific['llama_gc_tok_confidence'].tolist()
    specific_prob = []
    for c, g in zip(specific_confidences, specific_guesses):
        if 'yes' in g.lower():
            specific_prob.append(c)
        else:
            specific_prob.append(1 - c)
    df_test['specific_label'] = specific_prob
    df_generic = pd.read_csv('climretrieve_all_results/llama_gc_2e_generic.csv')
    generic_guesses = df_generic['llama_gc_guess'].tolist()
    generic_confidences = df_generic['llama_gc_tok_confidence'].tolist()
    generic_prob = []
    for c, g in zip(generic_confidences, generic_guesses):
        if 'yes' in g.lower():
            generic_prob.append(c)
        else:
            generic_prob.append(1 - c)
    df_test['generic_label'] = generic_prob
    all_questions = []
    for q in df_test['question']:
        if q in all_questions:
            continue
        else:
            all_questions.append(q)
    all_reports = []
    for q in df_test['report']:
        if q in all_reports:
            continue
        else:
            all_reports.append(q)
    for label_name in ['generic_label', 'specific_label']:
        for prob_name in ['bge_large_en', 'bge_large_en_ft', 'bge_base_en', 'bge_base_en_ft']:
            df_test['prob'] = df_test[prob_name]
            kendall_taus_gold_student = []
            for i, q in enumerate(all_questions):
                for j, r in enumerate(all_reports):
                    labels = df_test.loc[(df_test['question'] == q) & (df_test['report'] == r), label_name].tolist()
                    if len(labels) == 0:
                        continue
                    student_scores = df_test.loc[(df_test['question'] == q) & (df_test['report'] == r), 'prob'].tolist()
                    kendall_taus_gold_student.append(kendalltau(student_scores, labels, variant='b').statistic)

            kendall_tau_gold_student = float(np.mean(kendall_taus_gold_student))

            new_df = pd.DataFrame({
                "setting": label_name + '_' + prob_name,
                "kendall": round(kendall_tau_gold_student, 6),
            }, index=[0])
            if os.path.isfile('table5.xlsx'):
                df = pd.read_excel('table5.xlsx')
                df = pd.concat([df, new_df])
            else:
                df = new_df

            df.to_excel('table5.xlsx', index=False)


if __name__ == '__main__':
    table4()
    table5()
