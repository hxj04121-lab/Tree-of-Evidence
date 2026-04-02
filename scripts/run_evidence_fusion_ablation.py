import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing repo-root modules when running from scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evidence_tree import TreeOfEvidence  # noqa: E402
from generator import initial_generator  # noqa: E402
from searcher import RetrivalModel  # noqa: E402
from evaluate_aiops_anomaly_detection import (  # noqa: E402
    Confusion,
    _extract_pred_label,
    _normalize_gold,
)

from run_hdfs_aiops_chimera_cder_driver import _assert_openai_server_ready  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_or_empty(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return _load_json(path) or []


def _trim_invalid_tail(results: List[Dict[str, Any]], reasoning: List[Dict[str, Any]]) -> None:
    n = min(len(results), len(reasoning))
    if len(results) != n:
        del results[n:]
    if len(reasoning) != n:
        del reasoning[n:]

    while results:
        resp = (results[-1].get("response") or "").strip()
        if resp and _extract_pred_label(resp) is not None:
            break
        results.pop()
        reasoning.pop()


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


def evaluate_metrics(result_path: Path, test_path: Path) -> Metrics:
    results = _load_json(result_path)
    test = _load_json(test_path)
    n = min(len(results), len(test))
    if n <= 0:
        raise RuntimeError(f"Empty result/test overlap: {result_path} vs {test_path}")
    confusion = Confusion()
    for i in range(n):
        y_true = _normalize_gold(test[i].get("label"))
        if y_true is None:
            raise ValueError(f"Bad gold label at index {i}: {test[i].get('label')!r}")
        y_pred = _extract_pred_label((results[i].get("response") or "").strip())
        confusion.update(y_true, y_pred)
    return Metrics(
        samples=n,
        tp=confusion.tp,
        fp=confusion.fp,
        tn=confusion.tn,
        fn=confusion.fn,
        unknown=confusion.unknown,
    )


def _avg_fusion_calls(reason_path: Path) -> Optional[float]:
    items = _load_or_empty(reason_path)
    vals: List[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        v = it.get("fusion_llm_calls", None)
        if v is None:
            continue
        try:
            vals.append(int(v))
        except Exception:
            continue
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _infer_dataset_tag(test_file: Path) -> str:
    b = test_file.name.lower()
    if "bgl" in b:
        return "BGL"
    if "thunderbird" in b or "tbird" in b:
        return "TB"
    if "hdfs" in b:
        return "HDFS"
    return "AIOPS"


def _select_test_subset(rows_all: List[Dict[str, Any]], *, kind: str, n: int) -> List[Dict[str, Any]]:
    kind = str(kind or "").strip().lower()
    if n <= 0:
        return []
    if kind == "hdfs_seq":
        hdfs_re = re.compile(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+")
        out: List[Dict[str, Any]] = []
        for r in rows_all or []:
            if not isinstance(r, dict):
                continue
            if hdfs_re.search(r.get("question") or ""):
                out.append(r)
                if len(out) >= n:
                    break
        if len(out) < n:
            raise RuntimeError(f"Not enough HDFS seq questions in test file (need {n}, got {len(out)}).")
        return out

    # message (BGL/TB) or any other kind: take the first N rows.
    out = [r for r in (rows_all or []) if isinstance(r, dict)]
    if len(out) < n:
        raise RuntimeError(f"Not enough test rows (need {n}, got {len(out)}).")
    return out[:n]


def run_setting(
    *,
    tree_helper: TreeOfEvidence,
    args: argparse.Namespace,
    test_rows: List[Dict[str, Any]],
    out_path: Path,
    reason_path: Path,
    label: str,
) -> bool:
    started = time.time()
    results = _load_or_empty(out_path)
    reasoning = _load_or_empty(reason_path)
    _trim_invalid_tail(results, reasoning)

    flush_every = int(getattr(args, "flush_every", 10) or 10)
    flush_every = max(1, flush_every)

    i0 = len(results)
    if i0 > 0:
        print(f"[{label}] resume at {i0}/{len(test_rows)} -> {out_path}")
    else:
        print(f"[{label}] start {len(test_rows)} -> {out_path}")

    for i in range(i0, len(test_rows)):
        q = test_rows[i].get("question") or ""
        if not q:
            results.append({"question": "", "evidence": [], "documents": [], "response": ""})
            reasoning.append({"question": "", "mode": getattr(args, "retrieval_mode", "tot"), "nodes": []})
            continue

        response, reference, evidence, tree = tree_helper.aiops_chimera_cder(query=q)

        if not (response or "").strip():
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            raise RuntimeError(
                "Empty LLM response. The OpenAI-compatible server may be down. Restart it and rerun to resume."
            )
        if _extract_pred_label(str(response)) is None:
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            preview = str(response).strip().splitlines()[0].strip() if str(response).strip() else ""
            raise RuntimeError(f"Invalid response format (cannot parse label): {preview!r}")

        results.append({"question": q, "evidence": evidence, "documents": reference, "response": response})
        reasoning.append(tree)

        if (i + 1) % flush_every == 0 or i + 1 == len(test_rows):
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            dur = time.time() - started
            speed = (i + 1 - i0) / max(dur, 1e-9)
            print(f"[{label}] progress {i+1}/{len(test_rows)} saved (speed={speed:.2f} it/s)")

    done = len(results) >= len(test_rows)
    return bool(done)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evidence fusion strategy ablation (direct/vote/two_stage) for aiops_chimera_cder.")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--out_root", type=str, default="result/evidence_fusion_ablation")

    ap.add_argument("--aiops_query_kind", type=str, default="hdfs_seq", choices=["hdfs_seq", "message"])
    ap.add_argument("--test_file", type=str, default="data/aiops_eval/HDFS_labeled_test_2000_seq8.json")
    ap.add_argument("--corpus_tsv", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_corpus_mini.tsv")
    ap.add_argument("--embedding_pkl", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_emb_mini.pkl")

    # Model / prompt config (keep aligned across strategies).
    ap.add_argument("--generator_model", type=str, default="Qwen2.5-7B-Instruct")
    ap.add_argument("--thought_config_path", type=str, default="prompts/aiops_thought_prompt.json")
    ap.add_argument("--response_config_path", type=str, default="prompts/aiops_response_prompt_v8_hdfs_query_aware.json")

    # Retriever config
    ap.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--retriever_cache", type=str, default="./models")
    ap.add_argument("--retriever_device", type=str, default="cuda:0")
    ap.add_argument("--embedding_device", type=str, default="cuda:0")

    # Core knobs (kept constant across strategies)
    ap.add_argument("--top_k_documents", type=int, default=14)
    ap.add_argument("--block_chain_pool", type=int, default=2000)
    ap.add_argument("--block_chain_min_similarity", type=float, default=0.35)
    ap.add_argument("--aiops_other_block_select", type=str, default="kw_boost", choices=["freq", "max_score", "kw_boost"])
    ap.add_argument("--aiops_kw_boost_alpha", type=float, default=0.2)
    ap.add_argument("--aiops_localizer_kind", type=str, default="llm", choices=["heuristic", "llm"])
    ap.add_argument("--aiops_localizer_prompt_path", type=str, default="prompts/aiops_localizer_prompt_v1.json")
    ap.add_argument("--aiops_localizer_model", type=str, default="")
    ap.add_argument("--aiops_localizer_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_thought_max_gen_len", type=int, default=256)
    ap.add_argument("--aiops_localizer_top_lines", type=int, default=3)
    ap.add_argument("--aiops_align_enabled", type=int, default=1)
    ap.add_argument("--aiops_align_min_covered_lines", type=int, default=None)
    ap.add_argument("--aiops_align_max_retry", type=int, default=1)
    ap.add_argument("--aiops_hdfs_severity_guardrail", type=int, default=0)
    ap.add_argument("--aiops_bgl_normal_guardrail", type=int, default=0)
    ap.add_argument("--aiops_skip_thought", type=int, default=1, help="1=skip ToT (faster), 0=enable ToT loop.")

    # Parametric sweep: cross-block evidence count
    ap.add_argument("--other_blocks_values", type=str, default="0,3,6,9",
                     help="Comma-separated list of other_blocks values to sweep (e.g. '0,3,6,9').")

    ap.add_argument(
        "--strategies",
        type=str,
        default="direct,vote,two_stage",
        help="Comma-separated fusion strategies to run (subset of: direct,vote,two_stage). Use 'direct' to run only direct.",
    )

    # Generation runtime
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.1)
    ap.add_argument("--max_gen_len", type=int, default=256)
    ap.add_argument("--prompt_doc_max_chars", type=int, default=400)
    ap.add_argument("--prompt_query_max_chars", type=int, default=6000)
    ap.add_argument("--flush_every", type=int, default=10)

    args_cli = ap.parse_args()

    test_file = Path(args_cli.test_file)
    corpus_tsv = Path(args_cli.corpus_tsv)
    embedding_pkl = Path(args_cli.embedding_pkl)
    if not test_file.is_file():
        raise SystemExit(f"test_file not found: {test_file}")
    if not corpus_tsv.is_file():
        raise SystemExit(f"corpus_tsv not found: {corpus_tsv}")
    if not embedding_pkl.is_file():
        raise SystemExit(f"embedding_pkl not found: {embedding_pkl}")

    # Environment defaults
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
    os.environ.setdefault("OPENAI_API_KEY", "local")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if api_base:
        _assert_openai_server_ready(api_base)

    stamp = (args_cli.stamp or "").strip()
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")

    out_root = Path(args_cli.out_root)
    out_dir = out_root / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = _infer_dataset_tag(test_file)
    samples = int(args_cli.samples)
    if samples <= 0:
        raise SystemExit("--samples must be > 0")

    test_rows_all = _load_json(test_file)
    test_rows = _select_test_subset(test_rows_all, kind=args_cli.aiops_query_kind, n=samples)
    subset_path = out_dir / "test_subset.json"
    _dump_json(subset_path, test_rows)

    # Build base args namespace for the core framework.
    args = argparse.Namespace()

    # Retriever
    args.retriever = "gtr"
    args.retriever_model_name_or_path = args_cli.retriever_model_name_or_path
    args.retriever_type = "bert-base-uncased"
    args.retriever_cache = args_cli.retriever_cache
    args.retriever_device = args_cli.retriever_device
    args.embedding_device = args_cli.embedding_device
    args.gtr_embedding = str(embedding_pkl)
    args.wiki_passage = str(corpus_tsv)
    args.bm25_sphere_index = None
    args.load_index_path = None
    args.save_index_n_shards = 128
    args.top_k_documents = int(args_cli.top_k_documents)

    # distributed (off)
    args.per_gpu_embedder_batch_size = 512
    args.local_rank = -1
    args.main_port = -1

    # Generator (OpenAI-compatible local server)
    args.generator = args_cli.generator_model
    args.generator_file_path = ""
    args.generator_tokenizer_path = ""
    args.temperature = float(args_cli.temperature)
    args.top_p = float(args_cli.top_p)
    args.top_k = 40
    args.max_seq_len = 2048
    args.max_gen_len = int(args_cli.max_gen_len)
    args.max_batch_size = 1

    # Prompt truncation
    args.prompt_doc_max_chars = int(args_cli.prompt_doc_max_chars)
    args.prompt_query_max_chars = int(args_cli.prompt_query_max_chars)

    # Prompts
    args.thought_config_path = args_cli.thought_config_path
    args.response_config_path = args_cli.response_config_path
    args.thought_shot = 0
    args.response_shot = 0
    args.max_depth = 3
    args.max_nodes = None
    args.failed_parse_file = None
    args.with_model_evidence = 0
    args.missing_evidence_shot = 0
    args.missing_evidence_config_path = None
    args.evidence_fusion_config_path = None

    # AIOps mode
    args.retrieval_mode = "aiops_chimera_cder"
    args.aiops_query_kind = str(args_cli.aiops_query_kind or "hdfs_seq").strip().lower()
    args.aiops_skip_thought = int(args_cli.aiops_skip_thought)
    args.aiops_localizer_top_lines = int(args_cli.aiops_localizer_top_lines)
    args.aiops_localizer_kind = str(args_cli.aiops_localizer_kind)
    args.aiops_localizer_prompt_path = str(args_cli.aiops_localizer_prompt_path)
    args.aiops_localizer_model = str(args_cli.aiops_localizer_model) if args_cli.aiops_localizer_model else None
    args.aiops_localizer_max_gen_len = int(args_cli.aiops_localizer_max_gen_len)
    args.aiops_thought_max_gen_len = int(args_cli.aiops_thought_max_gen_len)
    args.aiops_other_block_select = str(args_cli.aiops_other_block_select)
    args.aiops_kw_boost_alpha = float(args_cli.aiops_kw_boost_alpha)
    args.aiops_align_enabled = int(args_cli.aiops_align_enabled)
    args.aiops_align_min_covered_lines = args_cli.aiops_align_min_covered_lines
    args.aiops_align_max_retry = int(args_cli.aiops_align_max_retry)
    args.aiops_hdfs_severity_guardrail = int(args_cli.aiops_hdfs_severity_guardrail)
    args.aiops_bgl_normal_guardrail = int(args_cli.aiops_bgl_normal_guardrail)

    # Cross-block retrieval budget
    args.block_chain_other_blocks = 9
    args.block_chain_pool = int(args_cli.block_chain_pool)
    args.block_chain_min_similarity = float(args_cli.block_chain_min_similarity)
    args.block_chain_self_block_mode = "normal"
    args.aiops_other_blocks = None
    args.aiops_self_top_k = 5
    args.aiops_other_top_k = 1
    args.aiops_cross_append_mode = "all"

    # Misc / output
    args.flush_every = int(args_cli.flush_every)
    args.test_file_path = str(subset_path)
    args.quick_test_samples = samples
    args.output_file_path = ""
    args.reasoning_path_file = ""

    # Instantiate shared retriever/generator once.
    retriever = RetrivalModel(args)
    generator = initial_generator(args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=args)

    # Parse other_blocks sweep values.
    other_blocks_values = [int(x.strip()) for x in args_cli.other_blocks_values.split(",") if x.strip()]
    if not other_blocks_values:
        raise SystemExit("--other_blocks_values must contain at least one integer value")

    allowed = {"direct", "vote", "two_stage"}
    strategies = [s.strip() for s in str(args_cli.strategies or "").split(",") if s.strip()]
    if not strategies:
        raise SystemExit("--strategies must be non-empty")
    bad = [s for s in strategies if s not in allowed]
    if bad:
        raise SystemExit(f"Unsupported --strategies: {bad}. Allowed: {sorted(allowed)}")
    paths: Dict[str, Tuple[Path, Path]] = {}

    k_self = int(args.aiops_self_top_k)
    k_other = int(args.aiops_other_top_k)

    for st in strategies:
        for ob in other_blocks_values:
            tag = f"fusion_{st}_o{ob}"
            args.evidence_fusion_strategy = st
            args.block_chain_other_blocks = ob
            args.top_k_documents = k_self + ob * k_other
            out_path = out_dir / f"{tag}_results.json"
            reason_path = out_dir / f"{tag}_reasoning.json"
            paths[tag] = (out_path, reason_path)
            run_setting(
                tree_helper=tree_helper,
                args=args,
                test_rows=test_rows,
                out_path=out_path,
                reason_path=reason_path,
                label=f"{dataset_tag} {tag}",
            )

    # Metrics summary
    lines = [
        "# Evidence Fusion Strategy Ablation (Parametric)",
        "",
        f"- Stamp: `{stamp}`",
        f"- Dataset tag: `{dataset_tag}`",
        f"- Test subset: `{subset_path}`",
        f"- Corpus: `{corpus_tsv}`",
        f"- Embedding: `{embedding_pkl}`",
        f"- Mode: `aiops_chimera_cder` (aiops_skip_thought={args.aiops_skip_thought})",
        f"- Response prompt: `{args.response_config_path}`",
        f"- other_blocks values: `{other_blocks_values}`",
        f"- k_self={k_self}, k_other={k_other}",
        "",
        "| Strategy | other_blocks | max_docs | Samples | Precision | Recall | F1 | Unknown | Avg fusion LLM calls/sample | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for st in strategies:
        for ob in other_blocks_values:
            tag = f"fusion_{st}_o{ob}"
            out_path, reason_path = paths[tag]
            m = evaluate_metrics(out_path, subset_path)
            avg_calls = _avg_fusion_calls(reason_path)
            avg_calls_str = "" if avg_calls is None else f"{avg_calls:.2f}"
            max_docs = k_self + ob * k_other
            lines.append(
                f"| {st} | {ob} | {max_docs} | {m.samples} | {m.precision*100:.2f}% | {m.recall*100:.2f}% | {m.f1*100:.2f}% | {m.unknown} | {avg_calls_str} | `{out_path}` |"
            )

    summary = out_dir / "metrics_summary.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("DONE")
    print(f"Out dir: {out_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

