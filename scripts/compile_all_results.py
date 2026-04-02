"""
compile_all_results.py — Scan result directories and produce consolidated tables.

Scans result/ for all result JSON files, evaluates P/R/F1 against test files,
and outputs a unified markdown table. Also integrates bootstrap CI and cost
data if available.

Usage:
  python scripts/compile_all_results.py \
    --result_root result/ \
    --test_dir    data/aiops_eval/ \
    --output      result/compiled_tables/all_results.md
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate.anomaly_detection import _extract_pred_label, _normalize_gold


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _detect_dataset(path: str) -> Optional[str]:
    name = path.lower()
    if "bgl" in name:
        return "BGL"
    if "hdfs" in name:
        return "HDFS"
    if "thunderbird" in name or "tbird" in name:
        return "TB"
    return None


def _detect_test_file(dataset: str, test_dir: str, n_samples: int) -> Optional[str]:
    """Find the matching test JSON based on dataset and sample count."""
    candidates = []
    for fn in os.listdir(test_dir):
        if not fn.endswith(".json"):
            continue
        ds = _detect_dataset(fn)
        if ds != dataset:
            continue
        candidates.append(fn)

    # Prefer exact sample count match
    for fn in candidates:
        if f"_{n_samples}" in fn or f"_test_{n_samples}" in fn:
            return os.path.join(test_dir, fn)

    # Fallback: largest test file for this dataset
    if candidates:
        # Sort by number in filename (descending)
        def _extract_n(fn):
            nums = re.findall(r"(\d+)", fn)
            return max(int(x) for x in nums) if nums else 0
        candidates.sort(key=_extract_n, reverse=True)
        return os.path.join(test_dir, candidates[0])

    return None


def evaluate_result(result_path: str, test_path: str) -> Dict[str, Any]:
    """Compute P/R/F1 for a result file against a test file."""
    results = _load_json(result_path)
    test_data = _load_json(test_path)

    n = min(len(results), len(test_data))
    tp = fp = tn = fn_ = unknown = 0

    for i in range(n):
        gold = _normalize_gold(test_data[i].get("label", ""))
        if gold is None:
            continue
        pred = _extract_pred_label((results[i].get("response") or "").strip())
        if pred is None:
            unknown += 1
            pred = 0
        if gold == 1 and pred == 1: tp += 1
        elif gold == 0 and pred == 1: fp += 1
        elif gold == 0 and pred == 0: tn += 1
        elif gold == 1 and pred == 0: fn_ += 1

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn_) if (tp + fn_) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "samples": n,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn_,
        "unknown": unknown,
    }


def scan_results(result_root: str, test_dir: str) -> List[Dict[str, Any]]:
    """Scan all result JSONs and evaluate them."""
    rows = []
    result_root = Path(result_root)

    for jf in sorted(result_root.rglob("*result*.json")):
        # Skip reasoning files
        if "reasoning" in jf.name:
            continue

        try:
            data = _load_json(str(jf))
            if not isinstance(data, list) or len(data) == 0:
                continue
        except Exception:
            continue

        n_samples = len(data)
        dataset = _detect_dataset(str(jf))
        if not dataset:
            continue

        test_path = _detect_test_file(dataset, test_dir, n_samples)
        if not test_path:
            continue

        try:
            metrics = evaluate_result(str(jf), test_path)
        except Exception as e:
            print(f"  SKIP {jf}: {e}")
            continue

        # Infer method name from path
        rel = jf.relative_to(result_root)
        parts = list(rel.parts)
        method = "/".join(parts[:-1]) if len(parts) > 1 else parts[0].replace("_results.json", "").replace("result.json", "")

        rows.append({
            "method": method,
            "dataset": dataset,
            "path": str(jf),
            **metrics,
        })

    return rows


def write_markdown(rows: List[Dict[str, Any]], output: str) -> None:
    """Write consolidated markdown table."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # Group by dataset
    by_dataset: Dict[str, List] = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(r)

    with open(output, "w", encoding="utf-8") as f:
        f.write("# Compiled Experiment Results\n\n")
        f.write(f"Total result files: {len(rows)}\n\n")

        for ds in ["BGL", "HDFS", "TB"]:
            ds_rows = by_dataset.get(ds, [])
            if not ds_rows:
                continue

            f.write(f"## {ds}\n\n")
            f.write("| Method | Samples | P | R | F1 | TP | FP | TN | FN | Unk |\n")
            f.write("|--------|--------:|----:|----:|----:|---:|---:|---:|---:|----:|\n")

            # Sort by F1 descending
            ds_rows.sort(key=lambda x: x["f1"], reverse=True)
            for r in ds_rows:
                f.write(
                    f"| {r['method']} | {r['samples']} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                    f"{r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {r['unknown']} |\n"
                )
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Compile all experiment results into tables")
    ap.add_argument("--result_root", default="result", help="Root directory to scan")
    ap.add_argument("--test_dir", default="data/aiops_eval", help="Directory with test JSON files")
    ap.add_argument("--output", default="result/compiled_tables/all_results.md")
    args = ap.parse_args()

    print(f"Scanning {args.result_root}...")
    rows = scan_results(args.result_root, args.test_dir)
    print(f"Found {len(rows)} result files with valid evaluations")

    write_markdown(rows, args.output)
    print(f"Table written to {args.output}")

    # Also write JSON for programmatic access
    json_out = args.output.replace(".md", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"JSON written to {json_out}")


if __name__ == "__main__":
    main()
