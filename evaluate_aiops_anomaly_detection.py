
import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional


_LABEL_RE = re.compile(
    r"(?im)^\s*(?:-?\s*)?(?:judgment|prediction|label|判定|判断|结论)\s*[:：]\s*(normal|anomaly|正常|异常)\b"
)
_CONCLUSION_RE = re.compile(
    r"(?i)(?:the\s+(?:answer|result|conclusion|classification)\s+is|"
    r"(?:overall|finally|therefore|thus|hence)\s*,?\s*(?:it\s+is\s+|this\s+is\s+)?|"
    r"classify\s+(?:it\s+)?as|classified?\s+as|"
    r"(?:this|the\s+(?:query|log(?:\s+sequence)?|sequence|case|sample|instance))\s+"
    r"(?:is|indicates|looks|appears|seems)\s+(?:a\s+|an\s+)?)"
    r"\s*[:：-]?\s*(normal|anomaly|正常|异常)\b"
)


def _normalize_gold(label: str) -> Optional[int]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"anomaly", "abnormal", "1", "true", "yes", "异常"}:
        return 1
    if s in {"normal", "0", "false", "no", "正常"}:
        return 0
    return None


def _extract_pred_label(response: str) -> Optional[int]:
    if not response:
        return None
    match = _LABEL_RE.search(response)
    if match:
        token = match.group(1).strip().lower()
        if token in {"anomaly", "异常"}:
            return 1
        if token in {"normal", "正常"}:
            return 0

    matches = list(_CONCLUSION_RE.finditer(response))
    if matches:
        token = matches[-1].group(1).strip().lower()
        if token in {"anomaly", "异常"}:
            return 1
        if token in {"normal", "正常"}:
            return 0

    low = response.lower()
    has_anomaly = "anomaly" in low or "异常" in response
    has_normal = "normal" in low or "正常" in response
    if has_anomaly and not has_normal:
        return 1
    if has_normal and not has_anomaly:
        return 0
    return None


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    unknown: int = 0

    def update(self, y_true: int, y_pred: Optional[int]) -> None:
        if y_pred is None:
            self.unknown += 1
            y_pred = 0
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        elif y_true == 1 and y_pred == 0:
            self.fn += 1

    def precision(self) -> float:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def evaluate(result_path: str, labeled_test_path: str, dump_errors: Optional[str] = None) -> None:
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(labeled_test_path, "r", encoding="utf-8") as f:
        test = json.load(f)

    n = min(len(results), len(test))
    if n == 0:
        raise RuntimeError("Empty result/test file (or no overlap).")

    confusion = Confusion()
    errors = []

    for i in range(n):
        gold_raw = test[i].get("label")
        y_true = _normalize_gold(gold_raw)
        if y_true is None:
            raise ValueError(f"Bad gold label at index {i}: {gold_raw!r}")

        resp = (results[i].get("response") or "").strip()
        y_pred = _extract_pred_label(resp)
        confusion.update(y_true, y_pred)

        if y_pred is None or y_pred != y_true:
            errors.append(
                {
                    "idx": i,
                    "question": test[i].get("question"),
                    "gold": "Anomaly" if y_true == 1 else "Normal",
                    "pred": None if y_pred is None else ("Anomaly" if y_pred == 1 else "Normal"),
                    "response": resp[:2000],
                    "meta": test[i].get("meta", {}),
                }
            )

    p = confusion.precision() * 100
    r = confusion.recall() * 100
    f1 = confusion.f1() * 100

    print("=" * 72)
    print(f"Result: {result_path}")
    print(f"Test:   {labeled_test_path}")
    print("-" * 72)
    print(f"Samples: {n}")
    print(f"TP/FP/TN/FN: {confusion.tp}/{confusion.fp}/{confusion.tn}/{confusion.fn}")
    print(f"Unknown predictions: {confusion.unknown}")
    print(f"Precision: {p:.2f}%")
    print(f"Recall:    {r:.2f}%")
    print(f"F1:        {f1:.2f}%")
    print("=" * 72)

    if dump_errors:
        with open(dump_errors, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(errors)} errors to: {dump_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AIOps anomaly detection (precision/recall/F1)")
    parser.add_argument("--result", required=True, help="Result JSON produced by main.py")
    parser.add_argument("--test", required=True, help="Labeled test JSON produced by adapt_logs.py")
    parser.add_argument("--dump_errors", default=None, help="Optional path to write misclassified cases as JSON")
    args = parser.parse_args()

    evaluate(result_path=args.result, labeled_test_path=args.test, dump_errors=args.dump_errors)


if __name__ == "__main__":
    main()
