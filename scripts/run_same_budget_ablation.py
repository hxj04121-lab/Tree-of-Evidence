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
    ap = argparse.ArgumentParser(description="Same-budget (single-pass vs ToT) ablation runner.")

    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="result")
    ap.add_argument("--dataset_tag", type=str, default="")

    ap.add_argument("--aiops_query_kind", type=str, default="message", choices=["hdfs_seq", "message"])
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--corpus_tsv", type=str, required=True)
    ap.add_argument("--embedding_pkl", type=str, required=True)

    ap.add_argument(
        "--retrieval_mode",
        type=str,
        default="aiops_block_chain_tot",
        choices=["aiops_block_chain_tot", "aiops_chimera_cder_tot"],
        help="ToT-style mode for the comparison (single-pass uses the same mode with --aiops_skip_thought=1).",
    )
    ap.add_argument("--gate_mode", type=str, default="full", choices=["full", "vr_only", "no_gate"])

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
    ap.add_argument("--aiops_thought_max_gen_len", type=int, default=512)
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
    base_args.retrieval_mode = str(args_cli.retrieval_mode or "aiops_block_chain_tot").strip()
    base_args.block_chain_other_blocks = int(args_cli.block_chain_other_blocks)
    base_args.aiops_self_top_k = int(args_cli.aiops_self_top_k)
    base_args.aiops_other_top_k = int(args_cli.aiops_other_top_k)
    base_args.aiops_cross_append_mode = str(args_cli.aiops_cross_append_mode or "all").strip().lower()
    base_args.gate_mode = str(args_cli.gate_mode or "full").strip().lower()

    retriever = RetrivalModel(base_args)
    generator = initial_generator(base_args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=base_args)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = _infer_dataset_tag(args_cli.test_file, explicit=args_cli.dataset_tag)
    n = len(test_rows)

    rows: List[Tuple[str, str]] = []
    all_done = True

    # A) ToT-style (thought loop enabled)
    base_args.aiops_skip_thought = 0
    tot_mode = str(base_args.retrieval_mode or "").strip()
    label_tot = f"{dataset_tag} {base_args.retrieval_mode} tot gate={base_args.gate_mode}"
    out_tot = str(out_dir / f"{dataset_tag}_same_budget_{kind}_{base_args.retrieval_mode}_tot_{n}_{stamp}.json")
    reason_tot = str(out_dir / f"{dataset_tag}_same_budget_{kind}_{base_args.retrieval_mode}_tot_{n}_{stamp}_reasoning.json")
    done = run_setting(tree_helper=tree_helper, args=base_args, test_rows=test_rows, out_path=out_tot, reason_path=reason_tot, label=label_tot)
    all_done = all_done and bool(done)
    rows.append((label_tot, out_tot))

    # --- Compute actual average retrieval budget from the ToT run ---
    avg_docs = None
    if os.path.isfile(reason_tot):
        try:
            reason_data = _load_json(reason_tot)
            doc_counts = []
            for entry in reason_data:
                if not isinstance(entry, dict):
                    continue
                # Count docs from block_groups (when available), otherwise approximate
                # total docs by (#retrieval rounds * docs_per_round).
                cost = entry.get("cost") or {}
                retrieval_events = ((cost.get("retrieval") or {}).get("events") or [])

                # Count total docs from block_groups (actual field name in many reasoning JSONs)
                meta = entry.get("block_groups_final") or entry.get("block_groups_initial") or []
                n_docs = sum(int(g.get("n_docs", 0)) for g in meta if isinstance(g, dict))

                # ToT-style runs may do multiple retrieval rounds; approximate total docs
                # consumed by the thought loop as (rounds * docs_per_round).
                seeds = entry.get("retrieval_seeds") or []
                if isinstance(seeds, list) and seeds:
                    try:
                        k_self = int(entry.get("k_self", base_args.aiops_self_top_k or 5) or 0)
                        k_other = int(entry.get("k_other", base_args.aiops_other_top_k or 1) or 0)
                        other_blocks = entry.get("other_blocks") or []
                        n_blocks = len(other_blocks) if isinstance(other_blocks, list) else 0
                        docs_per_round = max(0, k_self) + max(0, n_blocks) * max(0, k_other)
                        n_docs = max(int(n_docs), int(docs_per_round) * int(len(seeds)))
                    except Exception:
                        pass

                if n_docs > 0:
                    doc_counts.append(n_docs)
                elif retrieval_events:
                    # Fallback: estimate from retrieval event count * top_k
                    doc_counts.append(len(retrieval_events) * int(base_args.aiops_self_top_k or 5))
            if doc_counts:
                avg_docs = sum(doc_counts) / len(doc_counts)
                print(f"[same_budget] ToT avg docs/sample: {avg_docs:.1f} (from {len(doc_counts)} samples)")
        except Exception as e:
            print(f"[same_budget] WARNING: Could not parse ToT reasoning: {e}")

    # B) Single-pass (skip thought) — match budget to ToT average
    base_args.aiops_skip_thought = 1
    if avg_docs is not None and avg_docs > 0:
        matched_k = max(1, round(avg_docs))
        # IMPORTANT: in aiops_* modes, `top_k_documents` may not directly control the final evidence budget.
        # Use global `tot` mode for the single-pass control so `top_k_documents` is the true budget knob.
        base_args.retrieval_mode = "tot"
        base_args.top_k_documents = matched_k
        print(f"[same_budget] Setting single-pass retrieval_mode=tot top_k_documents={matched_k} (matched to ToT avg)")

    label_sp = f"{dataset_tag} tot single_pass (k={base_args.top_k_documents})"
    out_sp = str(out_dir / f"{dataset_tag}_same_budget_{kind}_{base_args.retrieval_mode}_single_pass_{n}_{stamp}.json")
    reason_sp = str(out_dir / f"{dataset_tag}_same_budget_{kind}_{base_args.retrieval_mode}_single_pass_{n}_{stamp}_reasoning.json")
    done = run_setting(tree_helper=tree_helper, args=base_args, test_rows=test_rows, out_path=out_sp, reason_path=reason_sp, label=label_sp)
    all_done = all_done and bool(done)
    rows.append((label_sp, out_sp))

    metrics_md = str(out_dir / f"{dataset_tag}_same_budget_{kind}_{tot_mode}_{n}_{stamp}.metrics.md")
    if all_done:
        write_metrics_md(rows, test_path=args_cli.test_file, out_md=metrics_md)
        print("DONE")
        print(f"Metrics: {metrics_md}")
    else:
        print("PARTIAL RUN (metrics not written). Rerun to continue.")


if __name__ == "__main__":
    main()
