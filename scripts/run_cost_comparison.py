import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, List, Tuple

# Allow importing repo-root modules when running from scripts/.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generator import initial_generator
from searcher import RetrivalModel
from evidence_tree import TreeOfEvidence

from run_hdfs_aiops_chimera_cder_driver import _assert_openai_server_ready, make_base_args, run_setting, write_metrics_md


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _infer_dataset_tag(test_file: str, explicit: str = "") -> str:
    if explicit:
        return explicit.strip()
    name = Path(test_file).name.lower()
    if "bgl" in name:
        return "BGL"
    if "thunderbird" in name or "tbird" in name:
        return "TB"
    if "hdfs" in name:
        return "HDFS"
    return "AIOPS"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cost comparison runner (single-pass RAG vs ToE no_gate vs ToE full).")

    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="result/cost_comparison")
    ap.add_argument("--dataset_tag", type=str, default="")

    ap.add_argument("--aiops_query_kind", type=str, default="message", choices=["hdfs_seq", "message"])
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--corpus_tsv", type=str, required=True)
    ap.add_argument("--embedding_pkl", type=str, required=True)

    # Base args required by make_base_args
    ap.add_argument("--generator_model", type=str, default="Qwen2.5-7B-Instruct")
    ap.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--retriever_cache", type=str, default="./models")
    ap.add_argument("--retriever_device", type=str, default="cuda:0")
    ap.add_argument("--embedding_device", type=str, default="cuda:0")

    ap.add_argument("--thought_config_path", type=str, default="prompts/aiops_thought_prompt.json")
    ap.add_argument("--response_config_path", type=str, default="prompts/aiops_response_prompt_v7_cite_evidence_grounded.json")

    ap.add_argument("--aiops_localizer_prompt_path", type=str, default="prompts/aiops_localizer_prompt_v1.json")
    ap.add_argument("--aiops_localizer_kind", type=str, default="llm", choices=["heuristic", "llm"])
    ap.add_argument("--aiops_localizer_model", type=str, default="")
    ap.add_argument("--aiops_localizer_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_thought_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_localizer_top_lines", type=int, default=3)

    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--top_k_documents", type=int, default=14)
    ap.add_argument("--block_chain_pool", type=int, default=2000)
    ap.add_argument("--block_chain_min_similarity", type=float, default=0.35)

    ap.add_argument("--block_chain_other_blocks", type=int, default=9)
    ap.add_argument("--aiops_self_top_k", type=int, default=5)
    ap.add_argument("--aiops_other_top_k", type=int, default=1)
    ap.add_argument("--aiops_cross_append_mode", type=str, default="all", choices=["best", "all"])

    ap.add_argument("--evidence_fusion_strategy", type=str, default="direct", choices=["direct", "vote", "two_stage"])
    ap.add_argument("--aiops_other_block_select", type=str, default="kw_boost", choices=["freq", "max_score", "kw_boost"])
    ap.add_argument("--aiops_kw_boost_alpha", type=float, default=0.2)

    ap.add_argument("--aiops_align_enabled", type=int, default=1)
    ap.add_argument("--aiops_align_min_covered_lines", type=int, default=None)
    ap.add_argument("--aiops_align_max_retry", type=int, default=1)
    ap.add_argument("--aiops_hdfs_severity_guardrail", type=int, default=0)
    ap.add_argument("--aiops_bgl_normal_guardrail", type=int, default=0)

    ap.add_argument("--aiops_dual_view_budget", type=int, default=14)
    ap.add_argument("--aiops_dual_view_k_private", type=int, default=10)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.1)
    ap.add_argument("--max_gen_len", type=int, default=96)
    ap.add_argument("--prompt_doc_max_chars", type=int, default=400)
    ap.add_argument("--prompt_query_max_chars", type=int, default=6000)
    ap.add_argument("--flush_every", type=int, default=10)
    ap.add_argument("--max_new", type=int, default=0)

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
    kind = str(args_cli.aiops_query_kind or "message").strip().lower()
    if kind == "hdfs_seq":
        hdfs_re = re.compile(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+")
        test_rows = [r for r in test_rows if hdfs_re.search(r.get("question") or "")]
        if not test_rows:
            raise RuntimeError("No HDFS seq questions found in the requested sample range.")
    else:
        if not test_rows:
            raise RuntimeError("No test rows found in the requested sample range.")

    base_args = make_base_args(args_cli)
    # Use aiops_chimera_cder so --aiops_skip_thought can toggle ToT vs single-pass.
    base_args.retrieval_mode = "aiops_chimera_cder"
    base_args.block_chain_other_blocks = int(args_cli.block_chain_other_blocks)
    base_args.aiops_self_top_k = int(args_cli.aiops_self_top_k)
    base_args.aiops_other_top_k = int(args_cli.aiops_other_top_k)
    base_args.aiops_cross_append_mode = str(args_cli.aiops_cross_append_mode or "all").strip().lower()
    base_args.evidence_fusion_strategy = str(args_cli.evidence_fusion_strategy or "direct").strip()

    retriever = RetrivalModel(base_args)
    generator = initial_generator(base_args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=base_args)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = _infer_dataset_tag(args_cli.test_file, explicit=args_cli.dataset_tag)
    n = len(test_rows)

    def out_paths(name: str) -> Tuple[str, str]:
        return (
            str(out_dir / f"{dataset_tag}_cost_cmp_{kind}_{name}_{n}_{stamp}.json"),
            str(out_dir / f"{dataset_tag}_cost_cmp_{kind}_{name}_{n}_{stamp}_reasoning.json"),
        )

    runs: List[Tuple[str, str, dict]] = [
        ("Single-pass RAG (skip_thought=1)", "single_pass", {"aiops_skip_thought": 1, "gate_mode": "full"}),
        ("ToE w/o gate (gate=no_gate)", "toe_no_gate", {"aiops_skip_thought": 0, "gate_mode": "no_gate"}),
        ("ToE full (gate=full)", "toe_full", {"aiops_skip_thought": 0, "gate_mode": "full"}),
    ]

    rows: List[Tuple[str, str]] = []
    all_done = True

    for label, name, overrides in runs:
        for k, v in overrides.items():
            setattr(base_args, k, v)
        out_path, reason_path = out_paths(name)
        done = run_setting(tree_helper=tree_helper, args=base_args, test_rows=test_rows, out_path=out_path, reason_path=reason_path, label=label)
        all_done = all_done and bool(done)
        rows.append((label, out_path))

    metrics_md = str(out_dir / f"{dataset_tag}_cost_cmp_{kind}_{n}_{stamp}.metrics.md")
    if all_done:
        write_metrics_md(rows, test_path=args_cli.test_file, out_md=metrics_md)
        print("DONE")
        print(f"Metrics: {metrics_md}")
    else:
        print("PARTIAL RUN (metrics not written). Rerun to continue.")


if __name__ == "__main__":
    main()
