import json
import re
import string
import sys
import argparse
from collections import Counter
import numpy as np

# --- Reuse core logic from evaluate/hotpot_evaluate.py ---

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

# --- Adaptation logic for log datasets ---

def extract_ground_truth_from_question(question, dataset_type):
    """
    Extract ground truth from the query.
    Assumes Question format: "Find logs with message: <LOG_CONTENT>..."
    """
    # Remove "Find logs with message: " prefix
    clean_q = re.sub(r"^Find logs with message:\s*", "", question, flags=re.IGNORECASE)

    # Remove trailing ellipsis "..."
    clean_q = re.sub(r"\.\.\.$", "", clean_q)

    if dataset_type.lower() == 'hdfs':
        # For HDFS, Block ID is the key identifier.
        # For Token F1 computation, we use the full message content.
        return clean_q.strip()
    elif dataset_type.lower() == 'bgl':
        # For BGL, use the full message content.
        return clean_q.strip()
    else:
        return clean_q.strip()

def evaluate_file(result_file, dataset_type):
    print(f"Evaluating {result_file} for {dataset_type}...")
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {result_file} not found.")
        return

    metrics = {'f1': [], 'prec': [], 'recall': [], 'em': []}

    for item in results:
        question = item.get('question', '')
        response = item.get('response', '')

        # 1. Extract ground truth
        ground_truth = extract_ground_truth_from_question(question, dataset_type)

        # 2. If response indicates missing information, score is 0
        if "information is missing" in response:
            metrics['f1'].append(0)
            metrics['prec'].append(0)
            metrics['recall'].append(0)
            metrics['em'].append(0)
            continue

        # 3. Compute scores
        f1, prec, recall = f1_score(response, ground_truth)
        em = exact_match_score(response, ground_truth)

        metrics['f1'].append(f1)
        metrics['prec'].append(prec)
        metrics['recall'].append(recall)
        metrics['em'].append(1 if em else 0)

    # 4. Summary output
    print(f"\n=== {dataset_type} Evaluation Results ===")
    print(f"Total Samples: {len(results)}")
    print(f"Average Precision: {np.mean(metrics['prec']) * 100:.2f}%")
    print(f"Average Recall:    {np.mean(metrics['recall']) * 100:.2f}%")
    print(f"Average F1 Score:  {np.mean(metrics['f1']) * 100:.2f}%")
    print(f"Exact Match (EM):  {np.mean(metrics['em']) * 100:.2f}%")
    print("======================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_result", type=str, default="result/HDFS_tot_result.json", help="Path to HDFS result file")
    parser.add_argument("--bgl_result", type=str, default="result/BGL_tot_result.json", help="Path to BGL result file")
    args = parser.parse_args()

    evaluate_file(args.hdfs_result, "HDFS")
    evaluate_file(args.bgl_result, "BGL")
