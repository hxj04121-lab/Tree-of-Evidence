import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing repo-root modules (evidence_tree.py, searcher.py, generator.py) when running from scripts/.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_tree import TreeOfEvidence, _select_extractive_log_response
from generator import initial_generator
from searcher import RetrivalModel


LABEL_RE = re.compile(r"(?im)^\s*(?:-?\s*)?(?:judgment|prediction|label|判定|判断|结论)\s*[:：]\s*(normal|anomaly|正常|异常)\b")
CONCLUSION_RE = re.compile(
    r"(?i)(?:the\s+(?:answer|result|conclusion|classification)\s+is|"
    r"(?:overall|finally|therefore|thus|hence)\s*,?\s*(?:it\s+is\s+|this\s+is\s+)?|"
    r"classify\s+(?:it\s+)?as|classified?\s+as|"
    r"(?:this|the\s+(?:query|log(?:\s+sequence)?|sequence|case|sample|instance))\s+"
    r"(?:is|indicates|looks|appears|seems)\s+(?:a\s+|an\s+)?)"
    r"\s*[:：-]?\s*(normal|anomaly|正常|异常)\b"
)


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_or_empty(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or []


def _assert_openai_server_ready(base_url: str) -> None:
    """Fail fast if the OpenAI-compatible server is not reachable.

    Without this guard, runs may silently write empty responses, which corrupts
    result files and breaks resume/metrics.
    """
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Bad status from OpenAI-compatible server: {resp.status}")
    except Exception as e:
        raise RuntimeError(f"Cannot reach OpenAI-compatible server at {url}: {e}") from e


def _trim_invalid_tail(results: List[Dict[str, Any]], reasoning: List[Dict[str, Any]]) -> None:
    """Keep results/reasoning aligned and drop trailing invalid items.

    When the generator server is down mid-run, responses can be saved as empty strings.
    Those should not be treated as completed samples for resume.
    """
    n = min(len(results), len(reasoning))
    if len(results) != n:
        del results[n:]
    if len(reasoning) != n:
        del reasoning[n:]

    while results:
        resp = (results[-1].get("response") or "").strip()
        if resp:
            break
        results.pop()
        reasoning.pop()


def _normalize_gold(label: Any) -> Optional[int]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"anomaly", "abnormal", "1", "true", "yes", "异常"}:
        return 1
    if s in {"normal", "0", "false", "no", "正常"}:
        return 0
    return None


def _extract_pred_label(resp: str) -> Optional[int]:
    if not resp:
        return None
    m = LABEL_RE.search(resp)
    if m:
        tok = m.group(1).strip().lower()
        if tok in {"anomaly", "异常"}:
            return 1
        if tok in {"normal", "正常"}:
            return 0
    matches = list(CONCLUSION_RE.finditer(resp))
    if matches:
        tok = matches[-1].group(1).strip().lower()
        if tok in {"anomaly", "异常"}:
            return 1
        if tok in {"normal", "正常"}:
            return 0
    low = (resp or "").lower()
    has_a = "anomaly" in low or "异常" in resp
    has_n = "normal" in low or "正常" in resp
    if has_a and not has_n:
        return 1
    if has_n and not has_a:
        return 0
    return None


@dataclass
class Metrics:
    samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    unknown: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def evaluate_metrics(result_path: str, test_path: str) -> Metrics:
    results = _load_json(result_path)
    test = _load_json(test_path)
    n = min(len(results), len(test))
    if n <= 0:
        raise RuntimeError(f"Empty result/test overlap: {result_path} vs {test_path}")
    tp = fp = tn = fn = unknown = 0
    for i in range(n):
        y_true = _normalize_gold(test[i].get("label"))
        if y_true is None:
            raise ValueError(f"Bad gold label at index {i}: {test[i].get('label')!r}")
        y_pred = _extract_pred_label((results[i].get("response") or "").strip())
        if y_pred is None:
            unknown += 1
            y_pred = 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    return Metrics(samples=n, tp=tp, fp=fp, tn=tn, fn=fn, unknown=unknown)


def make_base_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace()

    # Retriever
    args.retriever = "gtr"
    args.retriever_model_name_or_path = cli.retriever_model_name_or_path
    args.retriever_type = "bert-base-uncased"
    args.retriever_cache = cli.retriever_cache
    args.retriever_device = cli.retriever_device
    args.embedding_device = cli.embedding_device
    args.gtr_embedding = cli.embedding_pkl
    args.wiki_passage = cli.corpus_tsv
    args.bm25_sphere_index = None
    args.load_index_path = None
    args.save_index_n_shards = 128
    args.top_k_documents = int(cli.top_k_documents)

    # distributed (kept off)
    args.per_gpu_embedder_batch_size = 512
    args.local_rank = -1
    args.main_port = -1

    # Generator (OpenAI-compatible local server)
    args.generator = cli.generator_model
    args.generator_file_path = ""
    args.generator_tokenizer_path = ""
    args.temperature = float(cli.temperature)
    args.top_p = float(cli.top_p)
    args.top_k = 40
    args.max_seq_len = 2048
    args.max_gen_len = int(cli.max_gen_len)
    args.max_batch_size = 1

    # Prompt truncation to keep local LLM fast
    args.prompt_doc_max_chars = int(cli.prompt_doc_max_chars)
    args.prompt_query_max_chars = int(cli.prompt_query_max_chars)

    # General toggles
    args.log_response_mode = "line"
    args.enable_log_match_fastpath = 0
    args.normalize_log_query = 1
    args.use_extractive_response = 1
    args.flush_every = int(cli.flush_every)
    # Resume-friendly partial runs: limit number of *new* samples processed per setting.
    args.max_new = int(getattr(cli, "max_new", 0) or 0)

    # Prompts
    args.thought_config_path = cli.thought_config_path
    args.response_config_path = cli.response_config_path

    # Shots / ToT params
    args.thought_shot = 0
    args.response_shot = 0
    args.max_depth = int(cli.max_depth)
    args.max_nodes = None
    args.failed_parse_file = None
    args.with_model_evidence = 0
    args.missing_evidence_shot = 0
    args.missing_evidence_config_path = None
    args.evidence_fusion_config_path = None

    # Retrieval mode default (ensures block index builds)
    args.retrieval_mode = "aiops_chimera_cder"

    # Block-chain core parameters
    args.block_chain_other_blocks = 9
    args.block_chain_pool = int(cli.block_chain_pool)
    args.block_chain_min_similarity = float(cli.block_chain_min_similarity)
    args.block_chain_self_block_mode = "normal"

    # AIOps common
    args.aiops_query_kind = str(getattr(cli, "aiops_query_kind", "hdfs_seq") or "hdfs_seq").strip().lower()
    # Default behavior historically used skip_thought=1 for "tot" (global retrieval) baselines.
    # Allow overriding from CLI so we can run true ToT under retrieval_mode=tot when needed.
    cli_skip = getattr(cli, "aiops_skip_thought", None)
    args.aiops_skip_thought = 1 if (cli_skip is None) else int(cli_skip)
    args.aiops_localizer_top_lines = int(cli.aiops_localizer_top_lines)
    args.aiops_other_blocks = None
    args.aiops_self_top_k = 5
    args.aiops_other_top_k = 1
    args.aiops_cross_append_mode = "all"

    # Chimera-migrated SAL + CDA-lite
    args.aiops_localizer_kind = str(cli.aiops_localizer_kind)
    args.aiops_localizer_prompt_path = str(cli.aiops_localizer_prompt_path)
    args.aiops_localizer_model = str(cli.aiops_localizer_model) if cli.aiops_localizer_model else None
    args.aiops_localizer_max_gen_len = int(cli.aiops_localizer_max_gen_len)
    args.aiops_thought_max_gen_len = int(cli.aiops_thought_max_gen_len)
    args.aiops_other_block_select = str(cli.aiops_other_block_select)
    args.aiops_kw_boost_alpha = float(cli.aiops_kw_boost_alpha)
    args.aiops_align_enabled = int(cli.aiops_align_enabled)
    args.aiops_align_min_covered_lines = cli.aiops_align_min_covered_lines
    args.aiops_align_max_retry = int(cli.aiops_align_max_retry)
    args.aiops_hdfs_severity_guardrail = int(getattr(cli, "aiops_hdfs_severity_guardrail", 0) or 0)
    args.aiops_hdfs_normal_guardrail = int(getattr(cli, "aiops_hdfs_normal_guardrail", 0) or 0)
    args.aiops_bgl_normal_guardrail = int(getattr(cli, "aiops_bgl_normal_guardrail", 0) or 0)
    args.evidence_fusion_strategy = str(getattr(cli, "evidence_fusion_strategy", "direct") or "direct")

    # Chimera dual-view (feature-level)
    args.aiops_dual_view_budget = int(cli.aiops_dual_view_budget)
    args.aiops_dual_view_k_private = int(cli.aiops_dual_view_k_private)

    # log-match-only flags (keep off)
    args.log_match_skip_thought = 0
    args.block_chain_prefix_output = 0
    args.block_chain_prefix_output_extra_ge = 50
    args.block_chain_thunderbird_self_block_mode = "query"

    # output paths (per setting)
    args.output_file_path = ""
    args.reasoning_path_file = ""
    args.test_file_path = cli.test_file
    args.quick_test_samples = int(cli.samples)
    return args


def run_setting(
    tree_helper: TreeOfEvidence,
    args: argparse.Namespace,
    test_rows: List[Dict[str, Any]],
    out_path: str,
    reason_path: str,
    label: str,
) -> bool:
    started = time.time()
    results = _load_or_empty(out_path)
    reasoning = _load_or_empty(reason_path)
    _trim_invalid_tail(results, reasoning)

    flush_every = int(getattr(args, "flush_every", 5) or 5)
    flush_every = max(1, flush_every)

    i0 = len(results)
    if i0 > 0:
        print(f"[{label}] resume at {i0}/{len(test_rows)} -> {out_path}")
    else:
        print(f"[{label}] start {len(test_rows)} -> {out_path}")

    max_new = int(getattr(args, "max_new", 0) or 0)
    i1 = len(test_rows)
    if max_new > 0:
        i1 = min(i1, i0 + max_new)

    for i in range(i0, i1):
        q = test_rows[i].get("question") or ""
        if not q:
            results.append({"question": "", "evidence": [], "documents": [], "response": ""})
            reasoning.append({"question": "", "mode": getattr(args, "retrieval_mode", "tot"), "nodes": []})
            continue

        mode = str(getattr(args, "retrieval_mode", "tot") or "tot").strip().lower()
        if mode == "aiops_chimera_cder":
            response, reference, evidence, tree = tree_helper.aiops_chimera_cder(query=q)
        elif mode == "aiops_chimera_cder_tot":
            response, reference, evidence, tree = tree_helper.aiops_chimera_cder(query=q)
        elif mode == "aiops_chimera_dualview":
            response, reference, evidence, tree = tree_helper.aiops_chimera_dualview(query=q)
        elif mode == "aiops_block_chain_tot":
            response, reference, evidence, tree = tree_helper.aiops_tree_of_thought_with_cross_block(query=q)
        else:
            response, reference, evidence, tree = tree_helper.tree_of_thought_without_fusion(query=q)

        # If the server goes down mid-run, stop immediately to avoid writing empty responses.
        if not (response or "").strip():
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            raise RuntimeError(
                "Empty LLM response. The OpenAI-compatible server may be down. "
                "Restart it and rerun to resume."
            )
        if _extract_pred_label(str(response)) is None:
            preview = str(response).strip().splitlines()[0].strip() if str(response).strip() else ""
            print(f"[{label}] warning: unknown prediction kept at sample {i}: {preview!r}")

        response_full = _select_extractive_log_response(q, reference, response_mode="line")
        response_prefix = _select_extractive_log_response(q, reference, response_mode="prefix")
        results.append(
            {
                "question": q,
                "evidence": evidence,
                "documents": reference,
                "response": response,
                "response_mode": getattr(args, "log_response_mode", "line"),
                "response_full": response_full,
                "response_prefix": response_prefix,
            }
        )
        reasoning.append(tree)

        if (i + 1) % flush_every == 0 or i + 1 == i1 or i + 1 == len(test_rows):
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            dur = time.time() - started
            speed = (i + 1 - i0) / max(dur, 1e-9)
            print(f"[{label}] progress {i+1}/{len(test_rows)} saved (speed={speed:.2f} it/s)")

    # Ensure final flush for partial runs.
    if len(results) != len(test_rows):
        _dump_json(out_path, results)
        _dump_json(reason_path, reasoning)

    done = len(results) >= len(test_rows)
    if not done and max_new > 0:
        print(f"[{label}] partial run saved ({len(results)}/{len(test_rows)}). Rerun to continue.")
    return done


def write_metrics_md(rows: List[Tuple[str, str]], test_path: str, out_md: str) -> None:
    lines = [
        "# AIOps Chimera-migrated (inference-only)",
        "",
        f"- Test: `{test_path}`",
        "",
        "| Setting | Samples | Precision | Recall | F1 | Unknown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, result_path in rows:
        m = evaluate_metrics(result_path=result_path, test_path=test_path)
        lines.append(
            f"| {label} | {m.samples} | {m.precision*100:.2f}% | {m.recall*100:.2f}% | {m.f1*100:.2f}% | {m.unknown} |"
        )
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AIOps Chimera-migrated experiments in a single process (shared retriever).")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--aiops_query_kind", type=str, default="hdfs_seq", choices=["hdfs_seq", "message"])
    ap.add_argument("--dataset_tag", type=str, default="", help="Prefix for output files (e.g., HDFS/BGL/TB). If empty, inferred from test_file basename.")
    ap.add_argument("--test_file", type=str, default="data/aiops_eval/HDFS_labeled_test_2000_seq8.json")
    ap.add_argument("--corpus_tsv", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_corpus_mini.tsv")
    ap.add_argument("--embedding_pkl", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_emb_mini.pkl")
    ap.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--retriever_cache", type=str, default="./models")
    ap.add_argument("--retriever_device", type=str, default="cuda:0")
    ap.add_argument("--embedding_device", type=str, default="cuda:0")

    ap.add_argument("--generator_model", type=str, default="Qwen2.5-7B-Instruct")
    ap.add_argument("--thought_config_path", type=str, default="prompts/aiops_thought_prompt.json")
    # Default response prompt: HDFS query-aware v8 (improves recall on HDFS seq8 by leveraging query-level severity).
    ap.add_argument("--response_config_path", type=str, default="prompts/aiops_response_prompt_v8_hdfs_query_aware.json")

    ap.add_argument("--aiops_localizer_prompt_path", type=str, default="prompts/aiops_localizer_prompt_v1.json")
    ap.add_argument("--aiops_localizer_kind", type=str, default="llm", choices=["heuristic", "llm"])
    ap.add_argument("--aiops_localizer_model", type=str, default="")
    ap.add_argument("--aiops_localizer_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_thought_max_gen_len", type=int, default=256)
    ap.add_argument(
        "--aiops_skip_thought",
        type=int,
        default=None,
        help="Override the default skip_thought behavior. Use 0 for true ToT, 1 to bypass thought.",
    )

    # When enabled, ToT may directly answer Anomaly under weak evidence if the query has clear anomaly cues.
    # This is a recall-oriented policy toggle (kept off by default to preserve baselines).
    ap.add_argument("--aiops_tot_relaxed", type=int, default=0)

    ap.add_argument("--aiops_other_block_select", type=str, default="kw_boost", choices=["freq", "max_score", "kw_boost"])
    ap.add_argument("--aiops_kw_boost_alpha", type=float, default=0.2)

    ap.add_argument("--aiops_align_enabled", type=int, default=1)
    ap.add_argument("--aiops_align_min_covered_lines", type=int, default=None)
    ap.add_argument("--aiops_align_max_retry", type=int, default=1)
    ap.add_argument("--aiops_hdfs_severity_guardrail", type=int, default=0)
    ap.add_argument("--aiops_hdfs_normal_guardrail", type=int, default=0)
    ap.add_argument("--aiops_bgl_normal_guardrail", type=int, default=0)
    ap.add_argument(
        "--evidence_fusion_strategy",
        type=str,
        default="direct",
        choices=["direct", "vote", "two_stage"],
        help="Evidence fusion strategy for aiops_chimera_cder: direct (default), vote, or two_stage.",
    )
    ap.add_argument("--aiops_dual_view_budget", type=int, default=14)
    ap.add_argument("--aiops_dual_view_k_private", type=int, default=10)

    ap.add_argument("--aiops_localizer_top_lines", type=int, default=3)

    ap.add_argument("--max_depth", type=int, default=1)
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
    ap.add_argument("--only_tags", type=str, default="", help="Comma-separated setting tags to run (e.g., tot,cder_single,sal_cda_single,dual_view_single).")

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
    gen_name = (getattr(base_args, "generator", "") or "").lower()
    if "deepseek" in gen_name:
        from generator_deepseek import DeepSeekChatGPTModel
        generator = DeepSeekChatGPTModel(args=base_args)
    else:
        generator = initial_generator(base_args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=base_args)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = (args_cli.dataset_tag or "").strip()
    if not dataset_tag:
        b = Path(args_cli.test_file).name.lower()
        if "bgl" in b:
            dataset_tag = "BGL"
        elif "thunderbird" in b or "tbird" in b:
            dataset_tag = "TB"
        elif "hdfs" in b:
            dataset_tag = "HDFS"
        else:
            dataset_tag = "AIOPS"

    n = len(test_rows)
    settings = []

    def out_paths(tag: str) -> Tuple[str, str]:
        return (
            str(out_dir / f"{dataset_tag}_aiops_chimera_{tag}_{n}_{stamp}.json"),
            str(out_dir / f"{dataset_tag}_aiops_chimera_{tag}_{n}_{stamp}_reasoning.json"),
        )

    if kind == "hdfs_seq":
        settings.append((f"{dataset_tag} tot (global, k=14)", "tot", {"top_k_documents": 14}, out_paths("tot")))
        settings.append(
            (
                f"{dataset_tag} cder ToT (aiops_block_chain_tot) single (k=14)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_skip_thought": 0,
                },
                out_paths("cder_tot_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} cder ToT (aiops_block_chain_tot) cross (self5+other9)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_skip_thought": 0,
                },
                out_paths("cder_tot_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} cder ToT (aiops_block_chain_tot) cross (self10+other4)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 4,
                    "aiops_self_top_k": 10,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 10,
                    "aiops_skip_thought": 0,
                },
                out_paths("cder_tot_cross_self10_other4"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} cder (aiops_block_chain_tot) single (k=14)",
                "aiops_block_chain_tot",
                {"block_chain_other_blocks": 0, "aiops_self_top_k": 14, "aiops_other_top_k": 1, "top_k_documents": 14},
                out_paths("cder_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} cder (aiops_block_chain_tot) cross (self5+other9)",
                "aiops_block_chain_tot",
                {"block_chain_other_blocks": 9, "aiops_self_top_k": 5, "aiops_other_top_k": 1, "top_k_documents": 5},
                out_paths("cder_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} SAL-CDA single (align on)",
                "aiops_chimera_cder",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_align_enabled": 1,
                },
                out_paths("sal_cda_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera ToT single (other=0, align on)",
                "aiops_chimera_cder_tot",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_align_enabled": 1,
                    "aiops_skip_thought": 0,
                },
                out_paths("chimera_tot_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera dual-view single (private10+shared4, align on)",
                "aiops_chimera_dualview",
                {"aiops_dual_view_budget": 14, "aiops_dual_view_k_private": 10, "aiops_align_enabled": 1},
                out_paths("dual_view_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} SAL-CDA cross (align on)",
                "aiops_chimera_cder",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_align_enabled": 1,
                },
                out_paths("sal_cda_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera ToT cross (self5+other9, align on)",
                "aiops_chimera_cder_tot",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_align_enabled": 1,
                    "aiops_skip_thought": 0,
                },
                out_paths("chimera_tot_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} SAL-CDA single (no align)",
                "aiops_chimera_cder",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_align_enabled": 0,
                },
                out_paths("sal_cda_single_noalign"),
            )
        )
    else:
        # Message-query anomaly detection (BGL/TB): do retrieval+response (skip ToT DFS). Budget control stays k=14.
        align = int(getattr(args_cli, "aiops_align_enabled", 0) or 0)
        settings.append((f"{dataset_tag} tot (global, k=14, skip_thought)", "tot", {"top_k_documents": 14}, out_paths("tot")))
        settings.append(
            (
                f"{dataset_tag} block+tot ToT single (aiops_block_chain_tot, other=0)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_skip_thought": 0,
                },
                out_paths("blocktot_tot_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} block+tot ToT cross (aiops_block_chain_tot, self5+other9)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_skip_thought": 0,
                },
                out_paths("blocktot_tot_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} block+tot single (aiops_block_chain_tot, other=0)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                },
                out_paths("blocktot_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} block+tot cross (aiops_block_chain_tot, self5+other9)",
                "aiops_block_chain_tot",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                },
                out_paths("blocktot_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera ToT single (other=0, align={align})",
                "aiops_chimera_cder_tot",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_align_enabled": align,
                    "aiops_skip_thought": 0,
                },
                out_paths("chimera_tot_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera CDER single (other=0, align={align})",
                "aiops_chimera_cder",
                {
                    "block_chain_other_blocks": 0,
                    "aiops_self_top_k": 14,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 14,
                    "aiops_align_enabled": align,
                },
                out_paths("chimera_cder_single"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera ToT cross (self5+other9, align={align})",
                "aiops_chimera_cder_tot",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_align_enabled": align,
                    "aiops_skip_thought": 0,
                },
                out_paths("chimera_tot_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera CDER cross (self5+other9, align={align})",
                "aiops_chimera_cder",
                {
                    "block_chain_other_blocks": 9,
                    "aiops_self_top_k": 5,
                    "aiops_other_top_k": 1,
                    "top_k_documents": 5,
                    "aiops_align_enabled": align,
                },
                out_paths("chimera_cder_cross"),
            )
        )
        settings.append(
            (
                f"{dataset_tag} Chimera dual-view (private10+shared4, align={align})",
                "aiops_chimera_dualview",
                {"aiops_dual_view_budget": 14, "aiops_dual_view_k_private": 10, "aiops_align_enabled": align},
                out_paths("chimera_dualview"),
            )
        )



    def _extract_tag_from_out_path(p: str) -> str:
        name = Path(p).name
        m = re.search(r"^[A-Za-z0-9]+_aiops_chimera_(.+?)_\d+_", name)
        return m.group(1) if m else name

    only = [t.strip() for t in (args_cli.only_tags or "").split(",") if t.strip()]
    if only:
        only_set = set(only)
        settings = [x for x in settings if _extract_tag_from_out_path(x[3][0]) in only_set]
        if not settings:
            raise RuntimeError(f"No settings matched --only_tags={args_cli.only_tags!r}")

    all_done = True
    for label, mode, overrides, (out_path, reason_path) in settings:
        base_args.retrieval_mode = mode
        for k, v in (overrides or {}).items():
            setattr(base_args, k, v)
        done = run_setting(
            tree_helper=tree_helper,
            args=base_args,
            test_rows=test_rows,
            out_path=out_path,
            reason_path=reason_path,
            label=label,
        )
        all_done = all_done and bool(done)

    if all_done:
        metrics_md = str(out_dir / f"{dataset_tag}_aiops_chimera_{kind}_{n}_{stamp}.metrics.md")
        write_metrics_md([(lbl, p[0]) for (lbl, _, _, p) in settings], test_path=args_cli.test_file, out_md=metrics_md)

        print("DONE")
        print(f"Metrics: {metrics_md}")
    else:
        print("PARTIAL RUN (metrics not written). Rerun to continue.")


if __name__ == "__main__":
    main()
