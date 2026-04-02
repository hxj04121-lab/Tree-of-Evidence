import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing repo-root modules when running from scripts/.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_tree import TreeOfEvidence
from generator import initial_generator
from searcher import RetrivalModel

# Reuse the existing single-process driver utilities (resume trimming, metrics, etc).
from run_hdfs_aiops_chimera_cder_driver import (
    _assert_openai_server_ready,
    _load_json,
    evaluate_metrics,
    make_base_args,
    run_setting,
    write_metrics_md,
)


def _parse_int_list(text: str) -> List[int]:
    s = str(text or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _mode_short(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m == "aiops_chimera_cder_tot":
        return "chimera_tot"
    if m == "aiops_chimera_cder":
        return "chimera_cder"
    if m == "aiops_block_chain_tot":
        return "blocktot_tot"
    if m == "tot":
        return "tot"
    m = re.sub(r"[^a-z0-9]+", "_", m)
    m = re.sub(r"_+", "_", m).strip("_")
    return m[:48] if m else "mode"


def _budget_max_docs(k_self: int, other_blocks: int, k_other: int) -> int:
    return int(max(1, int(k_self)) + max(0, int(other_blocks)) * max(0, int(k_other)))


@dataclass
class AblationSetting:
    tag: str
    label: str
    mode: str
    overrides: Dict[str, Any]
    out_path: str
    reason_path: str


def _infer_dataset_tag(dataset_tag: str, test_file: str) -> str:
    dataset_tag = (dataset_tag or "").strip()
    if dataset_tag:
        return dataset_tag
    b = Path(test_file).name.lower()
    if "bgl" in b:
        return "BGL"
    if "thunderbird" in b or "tbird" in b:
        return "TB"
    if "hdfs" in b:
        return "HDFS"
    return "AIOPS"


def _build_settings(
    *,
    dataset_tag: str,
    n: int,
    stamp: str,
    out_dir: Path,
    modes: List[str],
    rounds: List[str],
    # Depth round
    depth_values: List[int],
    depth_fixed_self_top_k: int,
    depth_fixed_other_blocks: int,
    depth_k_other: int,
    # Width round 1: self_top_k
    self_top_k_values: List[int],
    selfk_fixed_depth: int,
    selfk_fixed_other_blocks: int,
    selfk_k_other: int,
    # Width round 2: other_blocks
    other_blocks_values: List[int],
    other_fixed_depth: int,
    other_fixed_self_top_k: int,
    other_k_other: int,
) -> List[AblationSetting]:
    settings: List[AblationSetting] = []

    def out_paths(tag: str) -> Tuple[str, str]:
        return (
            str(out_dir / f"{dataset_tag}_aiops_chimera_{tag}_{n}_{stamp}.json"),
            str(out_dir / f"{dataset_tag}_aiops_chimera_{tag}_{n}_{stamp}_reasoning.json"),
        )

    rounds_set = {str(r).strip().lower() for r in (rounds or []) if str(r).strip()}
    if not rounds_set:
        rounds_set = {"depth", "self_top_k", "other_blocks"}

    for mode in modes:
        ms = _mode_short(mode)

        if "depth" in rounds_set:
            for d in depth_values:
                k_self = int(depth_fixed_self_top_k)
                other_blocks = int(depth_fixed_other_blocks)
                k_other = int(depth_k_other)
                max_docs = _budget_max_docs(k_self, other_blocks, k_other)
                tag = f"{ms}_ab_depth_d{d}_s{k_self}_o{other_blocks}"
                out_path, reason_path = out_paths(tag)
                settings.append(
                    AblationSetting(
                        tag=tag,
                        label=f"{dataset_tag} {ms} depth={d} (self_top_k={k_self}, other_blocks={other_blocks}, max_docs={max_docs})",
                        mode=mode,
                        overrides={
                            "max_depth": int(d),
                            "aiops_skip_thought": 0,
                            "aiops_self_top_k": k_self,
                            "aiops_other_top_k": k_other,
                            "block_chain_other_blocks": other_blocks,
                            # keep fallback budgets aligned with self budget
                            "top_k_documents": k_self,
                        },
                        out_path=out_path,
                        reason_path=reason_path,
                    )
                )

        if "self_top_k" in rounds_set:
            for k_self in self_top_k_values:
                k_self = int(k_self)
                other_blocks = int(selfk_fixed_other_blocks)
                k_other = int(selfk_k_other)
                d = int(selfk_fixed_depth)
                max_docs = _budget_max_docs(k_self, other_blocks, k_other)
                tag = f"{ms}_ab_selfk_d{d}_s{k_self}_o{other_blocks}"
                out_path, reason_path = out_paths(tag)
                settings.append(
                    AblationSetting(
                        tag=tag,
                        label=f"{dataset_tag} {ms} self_top_k={k_self} (depth={d}, other_blocks={other_blocks}, max_docs={max_docs})",
                        mode=mode,
                        overrides={
                            "max_depth": d,
                            "aiops_skip_thought": 0,
                            "aiops_self_top_k": k_self,
                            "aiops_other_top_k": k_other,
                            "block_chain_other_blocks": other_blocks,
                            "top_k_documents": k_self,
                        },
                        out_path=out_path,
                        reason_path=reason_path,
                    )
                )

        if "other_blocks" in rounds_set:
            for other_blocks in other_blocks_values:
                other_blocks = int(other_blocks)
                k_self = int(other_fixed_self_top_k)
                k_other = int(other_k_other)
                d = int(other_fixed_depth)
                max_docs = _budget_max_docs(k_self, other_blocks, k_other)
                tag = f"{ms}_ab_other_d{d}_s{k_self}_o{other_blocks}"
                out_path, reason_path = out_paths(tag)
                settings.append(
                    AblationSetting(
                        tag=tag,
                        label=f"{dataset_tag} {ms} other_blocks={other_blocks} (depth={d}, self_top_k={k_self}, max_docs={max_docs})",
                        mode=mode,
                        overrides={
                            "max_depth": d,
                            "aiops_skip_thought": 0,
                            "aiops_self_top_k": k_self,
                            "aiops_other_top_k": k_other,
                            "block_chain_other_blocks": other_blocks,
                            "top_k_documents": k_self,
                        },
                        out_path=out_path,
                        reason_path=reason_path,
                    )
                )

    return settings


def _write_metrics_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    cols = [
        "tag",
        "mode",
        "max_depth",
        "aiops_self_top_k",
        "aiops_other_top_k",
        "block_chain_other_blocks",
        "max_docs",
        "samples",
        "precision",
        "recall",
        "f1",
        "unknown",
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HDFS AIOps ablations (depth + width) in one process (shared retriever/generator).")

    # Shared options (kept close to run_hdfs_aiops_chimera_cder_driver.py for compatibility).
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--aiops_query_kind", type=str, default="hdfs_seq", choices=["hdfs_seq", "message"])
    ap.add_argument("--dataset_tag", type=str, default="")
    ap.add_argument("--test_file", type=str, default="data/aiops_eval/HDFS_labeled_test_2000_seq8.json")
    ap.add_argument("--corpus_tsv", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_corpus_mini.tsv")
    ap.add_argument("--embedding_pkl", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_emb_mini.pkl")
    ap.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--retriever_cache", type=str, default="./models")
    ap.add_argument("--retriever_device", type=str, default="cuda:0")
    ap.add_argument("--embedding_device", type=str, default="cuda:0")

    ap.add_argument("--generator_model", type=str, default="Qwen2.5-7B-Instruct")
    ap.add_argument("--thought_config_path", type=str, default="prompts/aiops_thought_prompt.json")
    ap.add_argument("--response_config_path", type=str, default="prompts/aiops_response_prompt_v7_cite_evidence_grounded.json")

    ap.add_argument("--aiops_localizer_prompt_path", type=str, default="prompts/aiops_localizer_prompt_v1.json")
    ap.add_argument("--aiops_localizer_kind", type=str, default="llm", choices=["heuristic", "llm"])
    ap.add_argument("--aiops_localizer_model", type=str, default="")
    ap.add_argument("--aiops_localizer_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_thought_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_tot_relaxed", type=int, default=0)
    ap.add_argument("--aiops_other_block_select", type=str, default="kw_boost", choices=["freq", "max_score", "kw_boost"])
    ap.add_argument("--aiops_kw_boost_alpha", type=float, default=0.2)
    ap.add_argument("--aiops_align_enabled", type=int, default=1)
    ap.add_argument("--aiops_align_min_covered_lines", type=int, default=None)
    ap.add_argument("--aiops_align_max_retry", type=int, default=1)
    ap.add_argument("--aiops_hdfs_severity_guardrail", type=int, default=0)
    ap.add_argument("--aiops_dual_view_budget", type=int, default=14)
    ap.add_argument("--aiops_dual_view_k_private", type=int, default=10)
    ap.add_argument("--aiops_localizer_top_lines", type=int, default=3)

    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--top_k_documents", type=int, default=14)
    ap.add_argument("--block_chain_pool", type=int, default=2000)
    ap.add_argument("--block_chain_min_similarity", type=float, default=0.35)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.1)
    ap.add_argument("--max_gen_len", type=int, default=96)

    ap.add_argument("--prompt_doc_max_chars", type=int, default=400)
    ap.add_argument("--prompt_query_max_chars", type=int, default=6000)

    ap.add_argument("--flush_every", type=int, default=10)
    ap.add_argument(
        "--max_new",
        type=int,
        default=0,
        help="Resume-friendly: process at most this many NEW samples per setting (0 = run to completion).",
    )
    ap.add_argument("--out_dir", type=str, default="result")

    # Ablation controls
    ap.add_argument(
        "--modes",
        type=str,
        default="aiops_chimera_cder_tot",
        help="Comma-separated retrieval modes (e.g., aiops_chimera_cder_tot,aiops_block_chain_tot).",
    )
    ap.add_argument(
        "--rounds",
        type=str,
        default="depth,self_top_k,other_blocks",
        help="Comma-separated rounds to run: depth,self_top_k,other_blocks.",
    )

    ap.add_argument("--depth_values", type=str, default="1,2,3,4,5")
    ap.add_argument("--depth_fixed_self_top_k", type=int, default=5)
    ap.add_argument("--depth_fixed_other_blocks", type=int, default=9)
    ap.add_argument("--depth_k_other", type=int, default=1)

    ap.add_argument("--self_top_k_values", type=str, default="1,3,5,7,10")
    ap.add_argument("--selfk_fixed_depth", type=int, default=3)
    ap.add_argument("--selfk_fixed_other_blocks", type=int, default=9)
    ap.add_argument("--selfk_k_other", type=int, default=1)

    ap.add_argument("--other_blocks_values", type=str, default="0,3,6,9,12")
    ap.add_argument("--other_fixed_depth", type=int, default=3)
    ap.add_argument("--other_fixed_self_top_k", type=int, default=5)
    ap.add_argument("--other_k_other", type=int, default=1)

    ap.add_argument(
        "--only_tags",
        type=str,
        default="",
        help="Comma-separated tags to run (matches the ablation tag in output filename).",
    )

    args_cli = ap.parse_args()

    if not os.path.isfile(args_cli.test_file):
        raise FileNotFoundError(args_cli.test_file)
    if not os.path.isfile(args_cli.corpus_tsv):
        raise FileNotFoundError(args_cli.corpus_tsv)
    if not os.path.isfile(args_cli.embedding_pkl):
        raise FileNotFoundError(args_cli.embedding_pkl)

    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if api_base:
        _assert_openai_server_ready(api_base)

    stamp = args_cli.stamp.strip() if args_cli.stamp else ""
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")

    test_rows_all = _load_json(args_cli.test_file)
    test_rows = list(test_rows_all[: int(args_cli.samples)])
    kind = str(args_cli.aiops_query_kind or "hdfs_seq").strip().lower()
    if kind == "hdfs_seq":
        hdfs_re = re.compile(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+")
        test_rows = [r for r in test_rows if hdfs_re.search(r.get("question") or "")]
        if not test_rows:
            raise RuntimeError("No HDFS seq questions found in the requested sample range.")
    else:
        if not test_rows:
            raise RuntimeError("No test rows found in the requested sample range.")

    base_args = make_base_args(args_cli)
    retriever = RetrivalModel(base_args)
    generator = initial_generator(base_args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=base_args)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = _infer_dataset_tag(args_cli.dataset_tag, args_cli.test_file)
    n = len(test_rows)

    modes = [m.strip() for m in str(args_cli.modes or "").split(",") if m.strip()]
    if not modes:
        raise RuntimeError("--modes must be non-empty")

    rounds = [r.strip() for r in str(args_cli.rounds or "").split(",") if r.strip()]

    depth_values = _parse_int_list(args_cli.depth_values)
    if not depth_values:
        raise RuntimeError("--depth_values is empty")
    self_top_k_values = _parse_int_list(args_cli.self_top_k_values)
    if not self_top_k_values:
        raise RuntimeError("--self_top_k_values is empty")
    other_blocks_values = _parse_int_list(args_cli.other_blocks_values)
    if not other_blocks_values:
        raise RuntimeError("--other_blocks_values is empty")

    settings = _build_settings(
        dataset_tag=dataset_tag,
        n=n,
        stamp=stamp,
        out_dir=out_dir,
        modes=modes,
        rounds=rounds,
        depth_values=depth_values,
        depth_fixed_self_top_k=int(args_cli.depth_fixed_self_top_k),
        depth_fixed_other_blocks=int(args_cli.depth_fixed_other_blocks),
        depth_k_other=int(args_cli.depth_k_other),
        self_top_k_values=self_top_k_values,
        selfk_fixed_depth=int(args_cli.selfk_fixed_depth),
        selfk_fixed_other_blocks=int(args_cli.selfk_fixed_other_blocks),
        selfk_k_other=int(args_cli.selfk_k_other),
        other_blocks_values=other_blocks_values,
        other_fixed_depth=int(args_cli.other_fixed_depth),
        other_fixed_self_top_k=int(args_cli.other_fixed_self_top_k),
        other_k_other=int(args_cli.other_k_other),
    )

    only = [t.strip() for t in str(args_cli.only_tags or "").split(",") if t.strip()]
    if only:
        only_set = set(only)
        settings = [s for s in settings if s.tag in only_set]
        if not settings:
            raise RuntimeError(f"No settings matched --only_tags={args_cli.only_tags!r}")

    plan_path = str(out_dir / f"{dataset_tag}_aiops_ablation_{kind}_{n}_{stamp}.plan.json")
    Path(plan_path).write_text(
        json.dumps(
            [
                {
                    "tag": s.tag,
                    "label": s.label,
                    "mode": s.mode,
                    "overrides": s.overrides,
                    "out_path": s.out_path,
                    "reason_path": s.reason_path,
                }
                for s in settings
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    all_done = True
    for s in settings:
        base_args.retrieval_mode = s.mode
        for k, v in (s.overrides or {}).items():
            setattr(base_args, k, v)
        done = run_setting(
            tree_helper=tree_helper,
            args=base_args,
            test_rows=test_rows,
            out_path=s.out_path,
            reason_path=s.reason_path,
            label=s.label,
        )
        all_done = all_done and bool(done)

    # Write metrics once everything is complete (same semantics as the baseline driver).
    metrics_md = str(out_dir / f"{dataset_tag}_aiops_ablation_{kind}_{n}_{stamp}.metrics.md")
    metrics_csv = str(out_dir / f"{dataset_tag}_aiops_ablation_{kind}_{n}_{stamp}.metrics.csv")
    if all_done:
        write_metrics_md([(s.label, s.out_path) for s in settings], test_path=args_cli.test_file, out_md=metrics_md)

        rows = []
        for s in settings:
            m = evaluate_metrics(result_path=s.out_path, test_path=args_cli.test_file)
            k_self = int(s.overrides.get("aiops_self_top_k", 0) or 0)
            k_other = int(s.overrides.get("aiops_other_top_k", 0) or 0)
            other_blocks = int(s.overrides.get("block_chain_other_blocks", 0) or 0)
            rows.append(
                {
                    "tag": s.tag,
                    "mode": s.mode,
                    "max_depth": int(s.overrides.get("max_depth", 0) or 0),
                    "aiops_self_top_k": k_self,
                    "aiops_other_top_k": k_other,
                    "block_chain_other_blocks": other_blocks,
                    "max_docs": _budget_max_docs(k_self, other_blocks, k_other),
                    "samples": m.samples,
                    "precision": round(m.precision, 6),
                    "recall": round(m.recall, 6),
                    "f1": round(m.f1, 6),
                    "unknown": m.unknown,
                }
            )
        _write_metrics_csv(rows, metrics_csv)

        print("DONE")
        print(f"Plan: {plan_path}")
        print(f"Metrics: {metrics_md}")
        print(f"Metrics (CSV): {metrics_csv}")
    else:
        print("PARTIAL RUN (metrics not written). Rerun to continue.")
        print(f"Plan: {plan_path}")


if __name__ == "__main__":
    main()
