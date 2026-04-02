import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    gold = gold[:len(prediction["answer"].keys())]
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        # if cur_id not in prediction['sp']:
        #     print('missing sp fact {}'.format(cur_id))
        #     can_eval_joint = False
        # else:
        #     sp_em, sp_prec, sp_recall = update_sp(
        #         metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        # if can_eval_joint:
        #     joint_prec = prec * sp_prec
        #     joint_recall = recall * sp_recall
        #     if joint_prec + joint_recall > 0:
        #         joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
        #     else:
        #         joint_f1 = 0.
        #     joint_em = em * sp_em
        #
        #     metrics['joint_em'] += joint_em
        #     metrics['joint_f1'] += joint_f1
        #     metrics['joint_prec'] += joint_prec
        #     metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)


def recall_score(prediction_file, gold_file):
    with open(prediction_file, "r") as f:
        predict = json.load(f)
    with open(gold_file, "r") as f:
        gold = json.load(f)

    gold = gold[:100]

    metric = {}
    for key, value in predict.items():
        recall = 0
        value = value[:len(gold)]
        assert len(value) == len(gold)
        for v, g in zip(value, gold):
            supported = g["supporting_facts"]
            context = g["context"]
            cdict = {c[0]: c[1] for c in context}
            cnt = 0
            for s in supported:
                title = s[0]
                text = cdict[title][s[1]]
                text = text.lower()
                v_title = [t.strip() for t in v["title"]]
                if title in v_title:
                    cnt += 1
                # for txt in v["text"]:
                #     txt = txt.lower()
                #     if text in txt:
                #         cnt += 1
                #         break
            recall += cnt/len(supported)
        recall /= len(gold)
        metric[key] = recall
    for key, value in metric.items():
        print("Method:{} Recall:{}".format(key, value))

    return


def recall_score_musique(prediction_file, gold_file):
    with open(prediction_file, "r") as f:
        predict = json.load(f)
    with open(gold_file, "r") as f:
        gold = json.load(f)

    gold = gold[:100]

    metric = {}
    for key, value in predict.items():
        recall = 0
        value = value[:len(gold)]
        assert len(value) == len(gold)
        for v, g in zip(value, gold):
            supported = g["question_decomposition"]
            cnt = 0
            for s in supported:
                document = g["paragraphs"][int(s["paragraph_support_idx"])]
                text = document["paragraph_text"].strip()
                title = document["title"].strip()
                text = text.lower()
                v_title = [t.strip() for t in v["title"]]
                if title in v_title:
                    cnt += 1
                # for txt in v["text"]:
                #     txt = txt.lower()
                #     if text in txt:
                #         cnt += 1
                #         break
            recall += cnt/len(supported)
        recall /= len(gold)
        metric[key] = recall
    for key, value in metric.items():
        print("Method:{} Recall:{}".format(key, value))


def case_study():
    compare_file = "./result/baselines/hotpotqa_gpt4/final/iter_retgen_gpt4_final.json"
    golden_file = "./data/hotpotqa/hotpot_dev_fullwiki_first_500.json"
    with open("./result/hotpotqa/final/hotpotqa-gpt4-contriever-shot1-final.json", "r") as f:
        toc_result = json.load(f)
    with open(compare_file, "r") as f:
        compare_result = json.load(f)
    with open(golden_file, "r") as f:
        golden = json.load(f)
    with open("./result/hotpotqa/evidence_tree/hotpotqa-gpt4-contriever-shot1-evidence-tree.json", "r") as f:
        toc_tree = json.load(f)
    with open("./result/hotpotqa/result/hotpotqa-gpt4-contriever-shot1.json", "r") as f:
        toc_evidence = json.load(f)
    with open("./result/baselines/hotpotqa_gpt4/result/iter_retgen_gpt4.json", "r") as f:
        compare_chain = json.load(f)

    golden = golden[: len(toc_result["answer"].keys())]
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    result = []
    for cnt, line in enumerate(golden):
        cur_id = line['_id']
        em_1, prec_1, recall_1 = update_answer(metrics, toc_result['answer'][cur_id], line['answer'])
        em_2, prec_2, recall_2 = update_answer(metrics, compare_result['answer'][cur_id], line['answer'])
        if recall_1 > 0.8 and recall_2 < 0.8:
            question = toc_tree[cnt]["question"]
            tor_path = toc_tree[cnt]["nodes"]
            tor_answer = toc_result['answer'][cur_id]
            compare_path = compare_chain[cnt]["path"]
            compare_answer = compare_result['answer'][cur_id]
            tmp = {
                "question": question,
                "tor_answer": tor_answer,
                "compare_answer": compare_answer,
                "ground_truth": golden[cnt]["answer"],
                "tor_path": tor_path,
                "tor_evidence": toc_evidence[cnt]["evidence"],
                "compare_path": compare_path,
            }
            result.append(tmp)
    with open("./result/case_study/hotpotqa_iter_retgen_case.json", "w") as f:
        json.dump(result, f)
    return




if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])
    # recall_score_musique(sys.argv[1], sys.argv[2])
    # case_study()