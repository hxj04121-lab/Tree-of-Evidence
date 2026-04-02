import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


@dataclass(frozen=True)
class Summary:
    path: str
    samples: int
    samples_with_cost: int
    missing_cost: int
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_latency_ms_total: float
    retrieval_wall_ms_total: float
    wall_time_ms_total: float


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_reasoning(reasoning_entries: List[Dict[str, Any]], path: str) -> Summary:
    samples = len(reasoning_entries or [])
    samples_with_cost = 0
    missing_cost = 0

    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_latency_ms_total = 0.0
    retrieval_wall_ms_total = 0.0
    wall_time_ms_total = 0.0

    for it in reasoning_entries or []:
        if not isinstance(it, dict):
            missing_cost += 1
            continue
        cost = it.get("cost")
        if not isinstance(cost, dict):
            missing_cost += 1
            continue
        samples_with_cost += 1

        wall_time_ms_total += _as_float(cost.get("wall_time_ms"), 0.0)

        retrieval = cost.get("retrieval") if isinstance(cost.get("retrieval"), dict) else {}
        retrieval_wall_ms_total += _as_float(retrieval.get("wall_ms_total"), 0.0)

        llm = cost.get("llm") if isinstance(cost.get("llm"), dict) else {}
        llm_total = llm.get("total") if isinstance(llm.get("total"), dict) else {}
        llm_calls += _as_int(llm_total.get("llm_calls"), 0)
        prompt_tokens += _as_int(llm_total.get("prompt_tokens"), 0)
        completion_tokens += _as_int(llm_total.get("completion_tokens"), 0)
        total_tokens += _as_int(llm_total.get("total_tokens"), 0)
        llm_latency_ms_total += _as_float(llm_total.get("latency_ms_total"), 0.0)

    return Summary(
        path=path,
        samples=samples,
        samples_with_cost=samples_with_cost,
        missing_cost=missing_cost,
        llm_calls=llm_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        llm_latency_ms_total=round(llm_latency_ms_total, 1),
        retrieval_wall_ms_total=round(retrieval_wall_ms_total, 1),
        wall_time_ms_total=round(wall_time_ms_total, 1),
    )


def _format_markdown(rows: List[Summary]) -> str:
    lines = [
        "| File | Samples | With cost | Missing | LLM calls | Total toks | LLM latency (ms) | Retrieval wall (ms) | Wall time (ms) | Avg toks | Avg LLM lat (ms) | Avg retrieval (ms) | Avg wall (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        denom = max(1, int(r.samples_with_cost) or 0)
        avg_toks = r.total_tokens / denom
        avg_llm_lat = r.llm_latency_ms_total / denom
        avg_ret = r.retrieval_wall_ms_total / denom
        avg_wall = r.wall_time_ms_total / denom
        lines.append(
            f"| {r.path} | {r.samples} | {r.samples_with_cost} | {r.missing_cost} | {r.llm_calls} | {r.total_tokens} | {r.llm_latency_ms_total:.1f} | {r.retrieval_wall_ms_total:.1f} | {r.wall_time_ms_total:.1f} | {avg_toks:.1f} | {avg_llm_lat:.1f} | {avg_ret:.1f} | {avg_wall:.1f} |"
        )
    return "\n".join(lines) + "\n"


def _format_tsv(rows: List[Summary]) -> str:
    header = "\t".join(
        [
            "file",
            "samples",
            "samples_with_cost",
            "missing_cost",
            "llm_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_latency_ms_total",
            "retrieval_wall_ms_total",
            "wall_time_ms_total",
            "avg_total_tokens",
            "avg_llm_latency_ms",
            "avg_retrieval_wall_ms",
            "avg_wall_time_ms",
        ]
    )
    lines = [header]
    for r in rows:
        denom = max(1, int(r.samples_with_cost) or 0)
        lines.append(
            "\t".join(
                [
                    r.path,
                    str(r.samples),
                    str(r.samples_with_cost),
                    str(r.missing_cost),
                    str(r.llm_calls),
                    str(r.prompt_tokens),
                    str(r.completion_tokens),
                    str(r.total_tokens),
                    f"{r.llm_latency_ms_total:.1f}",
                    f"{r.retrieval_wall_ms_total:.1f}",
                    f"{r.wall_time_ms_total:.1f}",
                    f"{(r.total_tokens / denom):.1f}",
                    f"{(r.llm_latency_ms_total / denom):.1f}",
                    f"{(r.retrieval_wall_ms_total / denom):.1f}",
                    f"{(r.wall_time_ms_total / denom):.1f}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-sample cost/latency from reasoning JSON files.")
    ap.add_argument(
        "--reasoning",
        type=str,
        nargs="+",
        default=None,
        help="One or more *reasoning.json paths (lists of per-sample reasoning dicts).",
    )
    ap.add_argument(
        "--reasoning_dir",
        type=str,
        nargs="+",
        default=None,
        help="One or more directories to scan recursively for *reasoning*.json files.",
    )
    ap.add_argument("--format", type=str, default="markdown", choices=["markdown", "tsv", "json"])
    ap.add_argument("--out", type=str, default="", help="Optional output path. If empty, print to stdout.")
    ap.add_argument("--output", type=str, default="", help="Alias of --out.")
    args = ap.parse_args()

    out_path = (args.output or args.out or "").strip()

    reasoning_paths: List[str] = []
    if args.reasoning:
        reasoning_paths.extend([str(p) for p in args.reasoning])
    if args.reasoning_dir:
        for d in args.reasoning_dir:
            root = Path(d)
            if not root.exists():
                raise SystemExit(f"--reasoning_dir not found: {root}")
            for p in root.rglob("*reasoning*.json"):
                if p.is_file():
                    reasoning_paths.append(p.as_posix())
    # De-dup but keep stable order.
    seen = set()
    reasoning_paths = [p for p in reasoning_paths if not (p in seen or seen.add(p))]

    if not reasoning_paths:
        raise SystemExit("Provide --reasoning and/or --reasoning_dir")

    rows: List[Summary] = []
    for p in reasoning_paths:
        path = Path(p)
        data = _load_json(path)
        if not isinstance(data, list):
            raise SystemExit(f"Reasoning file must be a JSON list: {path}")
        rows.append(summarize_reasoning(data, path.as_posix()))

    if args.format == "json":
        out_obj = [r.__dict__ for r in rows]
        text = json.dumps(out_obj, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "tsv":
        text = _format_tsv(rows)
    else:
        text = _format_markdown(rows)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
