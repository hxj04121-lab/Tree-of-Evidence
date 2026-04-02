import json
import logging
import copy
import math
import re
import os
import time as _time_mod

import dist_utils
import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HDFS_BLOCK_RE = re.compile(r"blk_-?\d+")
_QUERY_PREFIX_RE = re.compile(r"(?i)^\s*(?:\[query\]\s*)?find\s+logs\s+with\s+message\s*:\s*")
_SYSLOG_PREFIX_RE = re.compile(
    r"\b[A-Za-z0-9_.-]+\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"
)
_DOC_TEXT_LINE_RE = re.compile(r"(?im)^\s*text\s*:\s*(.*)\s*$")
_SYSLOG_PROGRAM_RE = re.compile(r"\b([A-Za-z0-9_.-]+)(?:\[\d+\])?:")


def doc_to_text(doc):
    return "title :" + doc['title'] + '\n' + "text :" + doc['text']


def _extract_log_target(question):
    raw = (question or "").strip()
    if not _QUERY_PREFIX_RE.search(raw):
        return None

    is_prefix_query = raw.rstrip().endswith("...")
    target = _QUERY_PREFIX_RE.sub("", raw).strip()
    if is_prefix_query and target.endswith("..."):
        target = target[:-3].strip()
    target = " ".join(target.split())
    # Some queries wrap the message in quotes, e.g. `"host Mon DD HH:MM:SS ...`.
    # Strip leading/trailing quotes so prefix matching can succeed against raw log lines.
    if target and target[0] in {"\"", "'"}:
        target = target[1:].lstrip()
    if target and target[-1] in {"\"", "'"}:
        target = target[:-1].rstrip()
    if not target:
        return None

    syslog_m = _SYSLOG_PREFIX_RE.search(target) or _SYSLOG_PREFIX_RE.search(raw)
    syslog_prefix = syslog_m.group(0) if syslog_m else None
    return {"target": target, "is_prefix_query": is_prefix_query, "syslog_prefix": syslog_prefix}


def _extract_doc_text_line(doc_text):
    if not doc_text:
        return ""
    m = _DOC_TEXT_LINE_RE.search(doc_text)
    if m:
        return (m.group(1) or "").strip()
    return str(doc_text).strip()


def _select_extractive_log_response(question, documents, response_mode="line"):
    parsed = _extract_log_target(question)
    if not parsed:
        return None

    target = parsed["target"]
    is_prefix_query = parsed["is_prefix_query"]
    syslog_prefix = parsed["syslog_prefix"]
    mode = str(response_mode).lower().strip() if response_mode is not None else "line"

    candidates = []
    for doc in documents or []:
        line = _extract_doc_text_line(doc)
        if not line:
            continue
        line_norm = " ".join(line.split())
        if syslog_prefix and syslog_prefix not in line_norm:
            continue

        if is_prefix_query:
            if line_norm.startswith(target):
                extra = max(0, len(line_norm) - len(target))
                candidates.append((extra, len(line_norm), line_norm, line))
        else:
            if line_norm == target:
                candidates.append((0, len(line_norm), line_norm, line))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    if is_prefix_query and mode == "prefix":
        return target
    if is_prefix_query and mode == "auto" and syslog_prefix:
        return target
    return candidates[0][3].strip()


def _extract_thunderbird_program(text: str) -> str:
    """
    Thunderbird syslog lines look like:
      <host> <Mon> <DD> <HH:MM:SS> <host>/<host> <program>[pid]: <msg>
    We use <program> as a lightweight component key for cross-host noise reduction.
    """
    s = " ".join((text or "").split())
    if not s:
        return ""
    parts = s.split()
    if len(parts) < 6:
        return ""
    tail = " ".join(parts[5:])
    m = _SYSLOG_PROGRAM_RE.search(tail)
    return (m.group(1).lower() if m else "")


def _maybe_truncate_to_target_prefix(args, query: str, documents_text: list, response: str) -> str:
    """
    In some log-match setups, the ground truth is the TARGET prefix (without the trailing ...).
    When using block_chain_tot, the model often copies the full matched line (longer), which hurts
    token-precision. This optional post-process truncates to TARGET, but only if:
      - the query is a prefix query
      - the response is grounded in Documents (exactly matches one document line after normalization)
      - the response line starts with TARGET
    """
    mode = int(getattr(args, "block_chain_prefix_output", 0) or 0)
    if mode <= 0:
        return response
    parsed = _extract_log_target(query)
    if not parsed or not parsed.get("is_prefix_query"):
        return response
    target = parsed.get("target") or ""
    if not target:
        return response
    resp = " ".join((response or "").split())
    if not resp or not resp.startswith(target):
        return response
    if mode == 2:
        extra_ge = int(getattr(args, "block_chain_prefix_output_extra_ge", 50) or 0)
        extra = len(resp) - len(target)
        if extra_ge > 0 and extra < extra_ge:
            return response

    # Grounding: response must equal some document "text :" line after normalization.
    for doc_text in documents_text or []:
        line = _extract_doc_text_line(doc_text)
        line_norm = " ".join((line or "").split())
        if not line_norm:
            continue
        if line_norm == resp and line_norm.startswith(target):
            return target
    return response

