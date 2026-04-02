import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
import sys

# Allow importing repo-root modules (evidence_tree.py, searcher.py, generator.py) when running from scripts/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Any, Dict, List, Optional, Tuple

from evidence_tree import TreeOfEvidence, _select_extractive_log_response
from generator import initial_generator
from searcher import RetrivalModel


HDFS_SEQ_HEADER_RE = re.compile(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+")
LABEL_RE = re.compile(r"(?im)^\s*(?:-?\s*)?(?:judgment|prediction|label|判定|判断|结论)\s*[:：]\s*(normal|anomaly|正常|异常)\b")
HDFS_BLOCK_RE = re.compile(r"blk_-?\d+")
CITE_RE = re.compile(r"\[(\d+)\]")


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


def extract_self_block_from_query(q: str) -> str:
    m = re.search(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+(blk_-?\d+)", q or "")
    return (m.group(1).lower() if m else "")


def doc_block_from_text(doc_text: str) -> str:
    m = HDFS_BLOCK_RE.search(doc_text or "")
    return (m.group(0).lower() if m else "")


def cited_doc_indices(resp: str) -> List[int]:
    out = []
    for m in CITE_RE.finditer(resp or ""):
        try:
            i = int(m.group(1))
        except Exception:
            continue
        if i > 0:
            out.append(i)
    seen = set()
    uniq = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def build_reason_map(reason_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not reason_path or not os.path.isfile(reason_path):
        return out
    rows = _load_json(reason_path)
    if isinstance(rows, list):
        for r in rows:
            q = r.get("question")
            if q:
                out[str(q)] = r
    return out


def cross_cite_rate(results: List[Dict[str, Any]], reasoning: Dict[str, Any]) -> Tuple[int, int, int]:
    cited_total = 0
    cited_cross = 0
    any_cross_rows = 0
    for r in results:
        q = r.get("question") or ""
        rr = reasoning.get(q, {})
        self_block = (rr.get("self_block") or "").strip().lower() or extract_self_block_from_query(q)
        cites = cited_doc_indices(r.get("response") or "")
        if not cites:
            continue
        row_has_cross = False
        docs = r.get("documents") or []
        for ci in cites:
            if ci - 1 < 0 or ci - 1 >= len(docs):
                continue
            cited_total += 1
            bid = doc_block_from_text(str(docs[ci - 1]))
            if bid and self_block and bid != self_block:
                cited_cross += 1
                row_has_cross = True
        if row_has_cross:
            any_cross_rows += 1
    return cited_total, cited_cross, any_cross_rows


def find_fixes(test: List[Dict[str, Any]], single: List[Dict[str, Any]], cross: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    fixes = []
    for i in range(min(len(test), len(single), len(cross))):
        y_true = _normalize_gold(test[i].get("label"))
        if y_true is None:
            continue
        s_pred = _extract_pred_label(single[i].get("response") or "")
        c_pred = _extract_pred_label(cross[i].get("response") or "")
        if s_pred == y_true:
            continue
        if c_pred != y_true:
            continue
        fixes.append(
            {
                "idx": i,
                "gold": "Anomaly" if y_true == 1 else "Normal",
                "question": test[i].get("question"),
                "single_shift": (single[i].get("response") or "")[:800],
                "cross_shift": (cross[i].get("response") or "")[:800],
            }
        )
        if len(fixes) >= n:
            break
    return fixes


def _assert_openai_server_ready(base_url: str) -> None:
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"Bad status from OpenAI-compatible server: {resp.status}")
    except Exception as e:
        raise RuntimeError(f"Cannot reach OpenAI-compatible server at {url}: {e}") from e


def make_base_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace()
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
    args.per_gpu_embedder_batch_size = 512
    args.local_rank = -1
    args.main_port = -1

    args.generator = cli.generator_model
    args.generator_file_path = ""
    args.generator_tokenizer_path = ""
    args.temperature = float(cli.temperature)
    args.top_p = float(cli.top_p)
    args.top_k = 40
    args.max_seq_len = 2048
    args.max_gen_len = int(cli.max_gen_len)
    args.max_batch_size = 1
    args.log_response_mode = "line"

    args.enable_log_match_fastpath = 0
    args.normalize_log_query = 1
    args.use_extractive_response = 1
    args.flush_every = int(cli.flush_every)

    args.retrieval_mode = "aiops_block_chain_tot"

    args.thought_config_path = cli.thought_config_path
    args.response_config_path = cli.response_config_path
    args.prompt_doc_max_chars = int(getattr(cli, "prompt_doc_max_chars", 0) or 0)
    args.prompt_query_max_chars = int(getattr(cli, "prompt_query_max_chars", 0) or 0)

    args.thought_shot = 0
    args.response_shot = 0
    args.max_depth = int(cli.max_depth)
    args.max_nodes = None
    args.failed_parse_file = None
    args.with_model_evidence = 0
    args.missing_evidence_shot = 0
    args.missing_evidence_config_path = None
    args.evidence_fusion_config_path = None

    args.block_chain_other_blocks = 9
    args.block_chain_pool = int(cli.block_chain_pool)
    args.block_chain_min_similarity = float(cli.block_chain_min_similarity)
    args.block_chain_self_block_mode = "normal"

    args.aiops_query_kind = "hdfs_seq"
    args.aiops_localizer_top_lines = int(cli.aiops_localizer_top_lines)
    args.aiops_other_blocks = None
    args.aiops_self_top_k = 5
    args.aiops_other_top_k = 1
    args.aiops_cross_append_mode = "all"
    args.aiops_skip_thought = 1

    args.log_match_skip_thought = 0
    args.block_chain_prefix_output = 0
    args.block_chain_prefix_output_extra_ge = 50
    args.block_chain_thunderbird_self_block_mode = "query"

    args.output_file_path = ""
    args.reasoning_path_file = ""
    args.test_file_path = cli.test_file
    args.quick_test_samples = int(cli.samples)
    return args


def run_setting(tree_helper: TreeOfEvidence, args: argparse.Namespace, test_rows: List[Dict[str, Any]], out_path: str, reason_path: str, label: str) -> None:
    started = time.time()
    results = _load_or_empty(out_path)
    reasoning = _load_or_empty(reason_path)

    flush_every = int(getattr(args, "flush_every", 5) or 5)
    if flush_every < 1:
        flush_every = 1

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

        mode = str(getattr(args, "retrieval_mode", "tot") or "tot").strip().lower()
        if mode == "aiops_block_chain_tot":
            response, reference, evidence, tree = tree_helper.aiops_tree_of_thought_with_cross_block(query=q)
        else:
            response, reference, evidence, tree = tree_helper.tree_of_thought_without_fusion(query=q)

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

        if (i + 1) % flush_every == 0 or i + 1 == len(test_rows):
            _dump_json(out_path, results)
            _dump_json(reason_path, reasoning)
            dur = time.time() - started
            speed = (i + 1 - i0) / max(dur, 1e-9)
            print(f"[{label}] progress {i+1}/{len(test_rows)} saved (speed={speed:.2f} it/s)")


def write_metrics_md(rows: List[Tuple[str, str]], test_path: str, out_md: str) -> None:
    lines = [
        "# HDFS AIOps cross-block proof (seq8)",
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


def write_analysis_txt(test_path: str, single_path: str, cross_path: str, single_shift_path: str, cross_shift_path: str, single_reason: str, cross_reason: str, single_shift_reason: str, cross_shift_reason: str, out_path: str, n_examples: int = 10) -> None:
    test = _load_json(test_path)
    single = _load_json(single_path)
    cross = _load_json(cross_path)
    single_shift = _load_json(single_shift_path)
    cross_shift = _load_json(cross_shift_path)

    reason_single = build_reason_map(single_reason)
    reason_cross = build_reason_map(cross_reason)
    reason_single_shift = build_reason_map(single_shift_reason)
    reason_cross_shift = build_reason_map(cross_shift_reason)

    def summarize(label: str, results: List[Dict[str, Any]], reason: Dict[str, Any]) -> List[str]:
        n = min(len(results), len(test))
        ok = 0
        unk = 0
        for i in range(n):
            y_true = _normalize_gold(test[i].get("label"))
            y_pred = _extract_pred_label(results[i].get("response") or "")
            if y_pred is None:
                unk += 1
                y_pred = 0
            if y_true is not None and y_pred == y_true:
                ok += 1
        cited_total, cited_cross, any_cross_rows = cross_cite_rate(results[:n], reason)
        out = [f"[{label}] samples={n} acc={ok/n:.1%} unknown={unk} ({unk/n:.1%})"]
        if cited_total:
            out.append(
                f"        cited_docs={cited_total} cross_cited={cited_cross} ({cited_cross/cited_total:.1%}) rows_with_cross_cite={any_cross_rows} ({any_cross_rows/n:.1%})"
            )
        else:
            out.append("        cited_docs=0")
        return out

    lines: List[str] = []
    lines.extend(summarize("SINGLE", single, reason_single))
    lines.extend(summarize("CROSS", cross, reason_cross))
    lines.extend(summarize("SINGLE_SHIFT", single_shift, reason_single_shift))
    lines.extend(summarize("CROSS_SHIFT", cross_shift, reason_cross_shift))
    lines.append("")
    lines.append(f"Examples where CROSS_SHIFT fixes SINGLE_SHIFT (first {n_examples}):")
    fixes = find_fixes(test, single_shift, cross_shift, n=n_examples)
    if not fixes:
        lines.append("  (none)")
    else:
        for ex in fixes:
            lines.append(f"  idx={ex['idx']} gold={ex['gold']}")
            lines.append(f"  question: {ex['question']}")
            lines.append(f"  single_shift: {ex['single_shift']}")
            lines.append(f"  cross_shift : {ex['cross_shift']}")
            lines.append("")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HDFS seq8 cross-block proof in a single process (shared retriever).")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--stamp", type=str, default="")
    ap.add_argument("--test_file", type=str, default="data/aiops_eval/HDFS_labeled_test_2000_seq8.json")
    ap.add_argument("--corpus_tsv", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_corpus_mini.tsv")
    ap.add_argument("--embedding_pkl", type=str, default="data/aiops_eval/mini_corpus_2000/HDFS_emb_mini.pkl")
    ap.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-large")
    ap.add_argument("--retriever_cache", type=str, default="./models")
    ap.add_argument("--retriever_device", type=str, default="cuda:0")
    ap.add_argument("--embedding_device", type=str, default="cuda:0")
    ap.add_argument("--generator_model", type=str, default="Qwen2.5-7B-Instruct")
    ap.add_argument("--thought_config_path", type=str, default="prompts/aiops_thought_prompt.json")
    ap.add_argument("--response_config_path", type=str, default="prompts/aiops_response_prompt_v4_cite.json")
    ap.add_argument("--max_depth", type=int, default=1)
    ap.add_argument("--top_k_documents", type=int, default=14)
    ap.add_argument("--block_chain_pool", type=int, default=2000)
    ap.add_argument("--block_chain_min_similarity", type=float, default=0.35)
    ap.add_argument("--aiops_localizer_top_lines", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.1)
    ap.add_argument("--max_gen_len", type=int, default=128)
    ap.add_argument("--prompt_doc_max_chars", type=int, default=400)
    ap.add_argument("--prompt_query_max_chars", type=int, default=6000)
    ap.add_argument("--flush_every", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="result")
    ap.add_argument("--analysis_examples", type=int, default=10)
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
    test_rows = [r for r in test_rows if HDFS_SEQ_HEADER_RE.search(r.get("question") or "")]
    if not test_rows:
        raise RuntimeError("No HDFS seq questions found in the requested sample range.")

    base_args = make_base_args(args_cli)
    retriever = RetrivalModel(base_args)
    generator = initial_generator(base_args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=base_args)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = 14
    n = len(test_rows)
    paths = {
        "TOT": (str(out_dir / f"HDFS_aiops_proof_tot_{n}_{stamp}.json"), str(out_dir / f"HDFS_aiops_proof_tot_{n}_{stamp}_reasoning.json")),
        "SINGLE": (str(out_dir / f"HDFS_aiops_proof_block_chain_single_{n}_{stamp}.json"), str(out_dir / f"HDFS_aiops_proof_block_chain_single_{n}_{stamp}_reasoning.json")),
        "CROSS": (str(out_dir / f"HDFS_aiops_proof_block_chain_cross_{n}_{stamp}.json"), str(out_dir / f"HDFS_aiops_proof_block_chain_cross_{n}_{stamp}_reasoning.json")),
        "SINGLE_SHIFT": (str(out_dir / f"HDFS_aiops_proof_block_chain_single_shift_{n}_{stamp}.json"), str(out_dir / f"HDFS_aiops_proof_block_chain_single_shift_{n}_{stamp}_reasoning.json")),
        "CROSS_SHIFT": (str(out_dir / f"HDFS_aiops_proof_block_chain_cross_shift_{n}_{stamp}.json"), str(out_dir / f"HDFS_aiops_proof_block_chain_cross_shift_{n}_{stamp}_reasoning.json")),
    }

    settings = [
        ("HDFS aiops tot (k=14)", "tot", {"top_k_documents": budget, "block_chain_other_blocks": 0, "aiops_self_top_k": budget, "aiops_other_top_k": 0, "block_chain_self_block_mode": "normal"}, paths["TOT"]),
        ("HDFS aiops cder single (k=14)", "aiops_block_chain_tot", {"top_k_documents": budget, "block_chain_other_blocks": 0, "aiops_self_top_k": budget, "aiops_other_top_k": 1, "block_chain_self_block_mode": "normal"}, paths["SINGLE"]),
        ("HDFS aiops cder cross (self5+other9)", "aiops_block_chain_tot", {"top_k_documents": 5, "block_chain_other_blocks": 9, "aiops_self_top_k": 5, "aiops_other_top_k": 1, "block_chain_self_block_mode": "normal"}, paths["CROSS"]),
        ("HDFS aiops cder single SHIFT", "aiops_block_chain_tot", {"top_k_documents": budget, "block_chain_other_blocks": 0, "aiops_self_top_k": budget, "aiops_other_top_k": 1, "block_chain_self_block_mode": "shift"}, paths["SINGLE_SHIFT"]),
        ("HDFS aiops cder cross SHIFT", "aiops_block_chain_tot", {"top_k_documents": 5, "block_chain_other_blocks": 9, "aiops_self_top_k": 5, "aiops_other_top_k": 1, "block_chain_self_block_mode": "shift"}, paths["CROSS_SHIFT"]),
    ]

    for label, mode, overrides, (out_path, reason_path) in settings:
        base_args.retrieval_mode = mode
        for k, v in (overrides or {}).items():
            setattr(base_args, k, v)
        run_setting(tree_helper=tree_helper, args=base_args, test_rows=test_rows, out_path=out_path, reason_path=reason_path, label=label)

    metrics_md = str(out_dir / f"HDFS_aiops_crossblock_proof_{n}_{stamp}.metrics.md")
    write_metrics_md([(lbl, p[0]) for (lbl, _, _, p) in settings], test_path=args_cli.test_file, out_md=metrics_md)

    analysis_txt = str(out_dir / f"HDFS_aiops_crossblock_proof_{n}_{stamp}.analysis.txt")
    write_analysis_txt(
        test_path=args_cli.test_file,
        single_path=paths["SINGLE"][0],
        cross_path=paths["CROSS"][0],
        single_shift_path=paths["SINGLE_SHIFT"][0],
        cross_shift_path=paths["CROSS_SHIFT"][0],
        single_reason=paths["SINGLE"][1],
        cross_reason=paths["CROSS"][1],
        single_shift_reason=paths["SINGLE_SHIFT"][1],
        cross_shift_reason=paths["CROSS_SHIFT"][1],
        out_path=analysis_txt,
        n_examples=int(args_cli.analysis_examples),
    )

    print("DONE")
    print(f"Metrics: {metrics_md}")
    print(f"Analysis: {analysis_txt}")


if __name__ == "__main__":
    main()




