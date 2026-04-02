"""
bootstrap_ci.py — Compute bootstrap 95% confidence intervals for P/R/F1.

Works on EXISTING result files without re-running experiments.
Resamples at the log-group level (each group = one sample).

Usage:
  python scripts/bootstrap_ci.py \
    --result_json result/some_experiment/results.json \
    --test_json   data/aiops_eval/HDFS_labeled_test_2000.json \
    --n_bootstrap 10000 \
    --seed 42

  # Batch mode: scan a directory of result files
  python scripts/bootstrap_ci.py \
    --result_dir result/evidence_fusion_ablation/ \
    --test_json  data/aiops_eval/HDFS_labeled_test_2000.json \
    --output     result/bootstrap_ci/summary.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate.anomaly_detection import _extract_pred_label, _normalize_gold


def _load_pairs(result_json: str, test_json: str) -> List[Tuple[int, Optional[int]]]:
    """Load (y_true, y_pred) pairs from result + test files."""
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(result_json, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    n = min(len(test_data), len(result_data))
    pairs = []
    for i in range(n):
        gold = _normalize_gold(test_data[i].get("label", ""))
        if gold is None:
            continue
        resp = (result_data[i].get("response") or "").strip()
        pred = _extract_pred_label(resp)
        # Treat unknown as negative (consistent with Confusion.update)
        if pred is None:
            pred = 0
        pairs.append((gold, pred))
    return pairs


def _compute_metrics(pairs: List[Tuple[int, int]]) -> Dict[str, float]:
    """Compute P/R/F1 from (y_true, y_pred) pairs."""
    tp = fp = tn = fn = 0
    for yt, yp in pairs:
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def bootstrap_ci(
    pairs: List[Tuple[int, int]],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for P/R/F1."""
    rng = np.random.RandomState(seed)
    n = len(pairs)
    if n == 0:
        return {}

    pairs_arr = np.array(pairs)  # shape (n, 2)

    boot_p = []
    boot_r = []
    boot_f1 = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sample = pairs_arr[idx]
        m = _compute_metrics([(int(row[0]), int(row[1])) for row in sample])
        boot_p.append(m["precision"])
        boot_r.append(m["recall"])
        boot_f1.append(m["f1"])

    alpha = 1 - ci_level
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    point = _compute_metrics(pairs)

    result = {}
    for name, boot_vals, point_val in [
        ("precision", boot_p, point["precision"]),
        ("recall", boot_r, point["recall"]),
        ("f1", boot_f1, point["f1"]),
    ]:
        vals = np.array(boot_vals)
        result[name] = {
            "point": round(point_val, 4),
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "ci_lo": round(float(np.percentile(vals, lo)), 4),
            "ci_hi": round(float(np.percentile(vals, hi)), 4),
            "ci_level": ci_level,
        }

    result["n_samples"] = n
    result["n_bootstrap"] = n_bootstrap
    return result


def format_metric(m: dict) -> str:
    """Format as: 0.8500 (95% CI: 0.8200-0.8800)"""
    return f"{m['point']:.4f} (95% CI: {m['ci_lo']:.4f}-{m['ci_hi']:.4f})"


def main():
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for P/R/F1")
    parser.add_argument("--result_json", default=None, help="Single result JSON file")
    parser.add_argument("--result_dir", default=None, help="Directory to scan for result JSONs")
    parser.add_argument("--test_json", required=True, help="Labeled test JSON")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output file (JSON or MD)")
    args = parser.parse_args()

    results = {}

    if args.result_json:
        pairs = _load_pairs(args.result_json, args.test_json)
        ci = bootstrap_ci(pairs, args.n_bootstrap, args.seed)
        results[args.result_json] = ci
        print(f"\n{args.result_json} ({len(pairs)} samples):")
        for metric in ("precision", "recall", "f1"):
            if metric in ci:
                print(f"  {metric}: {format_metric(ci[metric])}")

    elif args.result_dir:
        rdir = Path(args.result_dir)
        json_files = sorted(rdir.rglob("*result*.json"))
        if not json_files:
            json_files = sorted(rdir.rglob("*.json"))
        for jf in json_files:
            try:
                pairs = _load_pairs(str(jf), args.test_json)
                if not pairs:
                    continue
                ci = bootstrap_ci(pairs, args.n_bootstrap, args.seed)
                key = str(jf.relative_to(rdir))
                results[key] = ci
                print(f"\n{key} ({len(pairs)} samples):")
                for metric in ("precision", "recall", "f1"):
                    if metric in ci:
                        print(f"  {metric}: {format_metric(ci[metric])}")
            except Exception as e:
                print(f"  SKIP {jf}: {e}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        if args.output.endswith(".md"):
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("# Bootstrap Confidence Intervals\n\n")
                f.write(f"N_bootstrap={args.n_bootstrap}, seed={args.seed}\n\n")
                f.write("| Experiment | P | R | F1 | Samples |\n")
                f.write("|---|---|---|---|---|\n")
                for name, ci in results.items():
                    if "f1" not in ci:
                        continue
                    f.write(f"| {name} | {format_metric(ci['precision'])} | {format_metric(ci['recall'])} | {format_metric(ci['f1'])} | {ci.get('n_samples', '?')} |\n")
            print(f"\nSummary saved to {args.output}")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
