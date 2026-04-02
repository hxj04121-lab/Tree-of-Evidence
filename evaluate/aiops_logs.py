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

# --- Adaptation logic for AIOps output format ---

def extract_ground_truth_from_question(question):
    """
    Extract ground truth from query (consistent with original evaluation script).
    Assumes Question format: "Find logs with message: <LOG_CONTENT>..."
    """
    clean_q = re.sub(r"^Find logs with message:\s*", "", question, flags=re.IGNORECASE)
    clean_q = re.sub(r"\.\.\.$", "", clean_q)
    return clean_q.strip()

def extract_evidence_from_response(response):
    """
    Extract the "evidence" section from AIOps response format.
    Example format:
    Judgment: Anomaly
    Evidence: Reference logs show Block blk_-1608999687919862906 replication failed...
    Anomaly type: Network communication failure
    """
    # Try to extract the evidence line (supports both Chinese and English field names)
    evidence_match = re.search(r'证据[：:]\s*(.*?)(?=\n|异常类型|判定|$)', response, re.DOTALL)
    if evidence_match:
        evidence = evidence_match.group(1).strip()
        # Remove extra whitespace
        evidence = ' '.join(evidence.split())
        return evidence

    # If no evidence field found, try returning the entire response (backward compatible)
    if "判定：" not in response and "证据：" not in response:
        # Old format: return response content directly
        return response.strip()

    # If no evidence found, return empty string
    return ""

def extract_judgment_from_response(response):
    """
    Extract judgment (Normal/Anomaly) from response.
    """
    judgment_match = re.search(r'判定[：:]\s*(Normal|Anomaly)', response, re.IGNORECASE)
    if judgment_match:
        return judgment_match.group(1).strip()
    return None

def evaluate_file(result_file, dataset_type):
    print(f"\n{'='*60}")
    print(f"Evaluating {result_file} for {dataset_type}")
    print(f"{'='*60}")

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {result_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {result_file}")
        return

    if len(results) == 0:
        print(f"Warning: No samples to evaluate in {result_file}")
        return

    metrics = {'f1': [], 'prec': [], 'recall': [], 'em': []}
    judgment_stats = {'Normal': 0, 'Anomaly': 0, 'Unknown': 0}
    missing_info_count = 0

    for idx, item in enumerate(results):
        question = item.get('question', '')
        response = item.get('response', '')

        # 1. Extract ground truth from question
        ground_truth = extract_ground_truth_from_question(question)

        # 2. Extract evidence from response
        evidence = extract_evidence_from_response(response)

        # 3. Extract judgment
        judgment = extract_judgment_from_response(response)
        if judgment:
            judgment_stats[judgment] = judgment_stats.get(judgment, 0) + 1
        else:
            judgment_stats['Unknown'] += 1

        # 4. Handle missing information or empty evidence
        if "information is missing" in response.lower() or "信息缺失" in response or not evidence:
            missing_info_count += 1
            metrics['f1'].append(0)
            metrics['prec'].append(0)
            metrics['recall'].append(0)
            metrics['em'].append(0)
            continue

        # 5. Compute F1, Precision, Recall, EM (based on evidence vs ground truth matching)
        f1, prec, recall = f1_score(evidence, ground_truth)
        em = exact_match_score(evidence, ground_truth)

        metrics['f1'].append(f1)
        metrics['prec'].append(prec)
        metrics['recall'].append(recall)
        metrics['em'].append(1 if em else 0)

    # 6. Summary output
    total_samples = len(results)
    avg_f1 = np.mean(metrics['f1']) * 100 if metrics['f1'] else 0
    avg_prec = np.mean(metrics['prec']) * 100 if metrics['prec'] else 0
    avg_recall = np.mean(metrics['recall']) * 100 if metrics['recall'] else 0
    avg_em = np.mean(metrics['em']) * 100 if metrics['em'] else 0

    print(f"\nEvaluation Results:")
    print(f"  Total Samples:        {total_samples}")
    print(f"  Missing Info Count:   {missing_info_count}")
    print(f"  {'─'*42}")
    print(f"  Average Precision:    {avg_prec:.2f}%")
    print(f"  Average Recall:       {avg_recall:.2f}%")
    print(f"  Average F1 Score:     {avg_f1:.2f}%")
    print(f"  Exact Match (EM):     {avg_em:.2f}%")
    print(f"  {'─'*42}")
    print(f"\nJudgment Statistics:")
    print(f"  Normal:   {judgment_stats.get('Normal', 0)} ({judgment_stats.get('Normal', 0)/total_samples*100:.1f}%)")
    print(f"  Anomaly:  {judgment_stats.get('Anomaly', 0)} ({judgment_stats.get('Anomaly', 0)/total_samples*100:.1f}%)")
    print(f"  Unknown:  {judgment_stats.get('Unknown', 0)} ({judgment_stats.get('Unknown', 0)/total_samples*100:.1f}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AIOps log anomaly detection results")
    parser.add_argument("--hdfs_result", type=str, default="result/HDFS_aiops_result.json",
                        help="Path to HDFS result file")
    parser.add_argument("--bgl_result", type=str, default="result/BGL_aiops_result.json",
                        help="Path to BGL result file")
    parser.add_argument("--hdfs_only", action="store_true", help="Only evaluate HDFS results")
    parser.add_argument("--bgl_only", action="store_true", help="Only evaluate BGL results")

    args = parser.parse_args()

    if args.bgl_only:
        evaluate_file(args.bgl_result, "BGL")
    elif args.hdfs_only:
        evaluate_file(args.hdfs_result, "HDFS")
    else:
        evaluate_file(args.hdfs_result, "HDFS")
        evaluate_file(args.bgl_result, "BGL")
