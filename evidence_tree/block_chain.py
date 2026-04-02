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

from evidence_tree.utils import (
    doc_to_text,
    _extract_log_target,
    _extract_doc_text_line,
    _select_extractive_log_response,
    _extract_thunderbird_program,
    _maybe_truncate_to_target_prefix,
    _HDFS_BLOCK_RE,
    _QUERY_PREFIX_RE,
    _DOC_TEXT_LINE_RE,
)

logger = logging.getLogger(__name__)


class BlockChainMixin:
    def block_chain_retrieval(self, query):
        if not getattr(self.retriever, "block_index_enabled", False) or not hasattr(
            self.retriever, "get_gtr_documents_in_blocks"
        ):
            return self.tree_of_thought_without_fusion(query)

        raw = (query or "").strip()
        is_prefix_query = raw.rstrip().endswith("...")
        target = _QUERY_PREFIX_RE.sub("", raw).strip()
        if is_prefix_query and target.endswith("..."):
            target = target[:-3].strip()
        target = " ".join(target.split())

        m = _HDFS_BLOCK_RE.search(target) or _HDFS_BLOCK_RE.search(raw)
        if not m:
            return self.tree_of_thought_without_fusion(query)
        self_block = m.group(0)

        stripped = _HDFS_BLOCK_RE.sub("", target)
        stripped = " ".join(stripped.split()).strip()
        if not stripped:
            stripped = target

        max_other_blocks = int(getattr(self.args, "block_chain_other_blocks", 9))
        candidate_pool = int(getattr(self.args, "block_chain_pool", 2000))
        min_similarity = float(getattr(self.args, "block_chain_min_similarity", 0.35))
        early_stop_no_new = int(getattr(self.args, "block_chain_early_stop_no_new", 1))
        top_k = int(self.args.top_k_documents)

        device = self.args.embedding_device if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            q_ref_np = self.retriever.model.encode(
                [stripped], batch_size=1, show_progress_bar=False, normalize_embeddings=True
            )
            q_ref = torch.tensor(q_ref_np[0], dtype=torch.float16, device=device)

        global_docs = self.retriever.get_documents(question=[stripped], top_k=candidate_pool)[0]["documents"]
        best_by_block = {}
        for doc in global_docs:
            dm = _HDFS_BLOCK_RE.search(doc.get("text", "")) or _HDFS_BLOCK_RE.search(doc.get("title", ""))
            if not dm:
                continue
            bid = dm.group(0)
            if bid == self_block:
                continue
            prev = best_by_block.get(bid)
            if prev is None or doc["score"] > prev["score"]:
                best_by_block[bid] = doc

        other_blocks = [
            bid
            for bid, _ in sorted(best_by_block.items(), key=lambda kv: kv[1]["score"], reverse=True)[:max_other_blocks]
        ]
        block_ids = [self_block] + other_blocks

        depth_records = []
        answer_docs = []
        answer_seen = set()
        chain_selected_ids = {bid: set() for bid in block_ids}
        other_pool = []

        def _doc_global_index(doc_dict):
            return int(doc_dict["id"]) - 1

        def _ref_score(doc_dict):
            idx = _doc_global_index(doc_dict)
            emb = self.retriever.embedding[idx]
            return float(torch.matmul(emb, q_ref).item())

        def _record_evidence(doc_dict, meta, include_in_answer: bool):
            text = doc_to_text(doc_dict)
            added_to_answer = False
            if include_in_answer:
                if text in answer_seen:
                    return False
                answer_seen.add(text)
                answer_docs.append(text)
                added_to_answer = True
            meta = dict(meta)
            meta["doc_id"] = doc_dict.get("id")
            meta["title"] = doc_dict.get("title")
            meta["text"] = doc_dict.get("text")
            meta["ref_score"] = _ref_score(doc_dict)
            depth_records.append(meta)
            return added_to_answer

        no_new_streak = 0
        active_blocks = set(block_ids)

        for depth in range(1, int(self.args.max_depth) + 1):
            added_this_depth = False

            if depth == 1:
                queries = [target] + [stripped] * len(other_blocks)
                blocks = block_ids
            else:
                blocks = []
                queries = []
                for bid in list(active_blocks):
                    last_doc = chain_last.get(bid)
                    if last_doc is None:
                        continue
                    blocks.append(bid)
                    queries.append(last_doc["title"] + ". " + last_doc["text"])

            if not blocks:
                break

            responses = self.retriever.get_gtr_documents_in_blocks(queries, blocks, top_k=top_k)

            chain_last = {} if depth == 1 else chain_last
            current_other_candidates = []

            for bi, bid in enumerate(blocks):
                docs = responses[bi]["documents"]
                selected = None

                if bid == self_block and depth == 1:
                    matched = []
                    for d in docs:
                        if d["id"] in chain_selected_ids[bid]:
                            continue
                        text_norm = " ".join((d.get("text") or "").split())
                        if is_prefix_query:
                            if text_norm.startswith(target):
                                matched.append((len(text_norm) - len(target), len(text_norm), d))
                        else:
                            if text_norm == target:
                                matched.append((0, len(text_norm), d))

                    if not matched and getattr(self.retriever, "block_index_enabled", False):
                        blk_idx = (
                            self.retriever.block_to_idx.get(self_block)
                            if getattr(self.retriever, "block_to_idx", None) is not None
                            else None
                        )
                        offsets = getattr(self.retriever, "block_offsets", None)
                        doc_indices = getattr(self.retriever, "block_doc_indices", None)
                        if blk_idx is not None and offsets is not None and doc_indices is not None:
                            start = offsets[blk_idx]
                            end = offsets[blk_idx + 1]
                            for global_idx in doc_indices[start:end]:
                                title, text = self.retriever.docs[int(global_idx)].split("\n", 1)
                                text_norm = " ".join(text.split())
                                if is_prefix_query:
                                    if not text_norm.startswith(target):
                                        continue
                                else:
                                    if text_norm != target:
                                        continue
                                doc_dict = {
                                    "id": str(int(global_idx) + 1),
                                    "title": title,
                                    "text": text,
                                    "score": 0.0,
                                }
                                matched.append((len(text_norm) - len(target), len(text_norm), doc_dict))

                    if matched:
                        matched.sort(key=lambda t: (t[0], t[1]))
                        selected = matched[0][2]

                if selected is None:
                    for d in docs:
                        if d["id"] in chain_selected_ids[bid]:
                            continue
                        selected = d
                        break

                if selected is None:
                    active_blocks.discard(bid)
                    continue

                chain_selected_ids[bid].add(selected["id"])
                chain_last[bid] = selected

                if bid == self_block:
                    added_this_depth |= _record_evidence(
                        selected, {"depth": depth, "block": bid, "role": "self"}, include_in_answer=True
                    )
                else:
                    current_other_candidates.append((bid, selected))
                    other_pool.append((bid, selected))

            if current_other_candidates:
                best_bid, best_doc = max(current_other_candidates, key=lambda t: _ref_score(t[1]))
                best_score = _ref_score(best_doc)
                if best_score >= min_similarity:
                    # Noise-reduction: record cross-block evidence for analysis, but do NOT add it to the
                    # final "Documents" fed to the log-matching responder.
                    _record_evidence(best_doc, {"depth": depth, "block": best_bid, "role": "cross"}, include_in_answer=False)

            if not added_this_depth:
                no_new_streak += 1
            else:
                no_new_streak = 0

            if no_new_streak >= max(1, early_stop_no_new):
                break

        reasoning = {
            "mode": "block_chain",
            "self_block": self_block,
            "candidate_blocks": block_ids,
            "depth_records": depth_records,
        }

        response = _select_extractive_log_response(
            query, answer_docs, response_mode=getattr(self.args, "log_response_mode", "line")
        )
        if not int(getattr(self.args, "use_extractive_response", 1)):
            response = None
        if response is None:
            response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, answer_docs)
            response = self.generate("", response_prompt)
            if "gpt" in self.args.generator and response == "":
                response = self.retry_loop("", response_prompt)
        return response, answer_docs, depth_records, reasoning

    def _generate_with_model(self, model, system_prompt: str, inputs: str) -> str:
        if "gpt" in str(getattr(getattr(model, "args", None), "generator", "")).lower():
            body = model.get_body(system_prompt=system_prompt, input=inputs)
            return model.get_response(body=body)
        return model.get_response(queries=[inputs], system_prompt=system_prompt)

    def _generate_with_temp_max_len(self, model, system_prompt: str, inputs: str, max_gen_len: int) -> str:
        old_len = None
        try:
            if getattr(model, "args", None) is not None:
                old_len = getattr(model.args, "max_gen_len", None)
                model.args.max_gen_len = int(max_gen_len)
            return self._generate_with_model(model, system_prompt, inputs)
        finally:
            try:
                if old_len is not None and getattr(model, "args", None) is not None:
                    model.args.max_gen_len = old_len
            except Exception:
                pass

    def block_chain_vote_retrieval(self, query):
        """
        Cross-block retrieval + per-block grounded candidates + LLM vote (no ToT DFS).

        Output format matches other modes: (response, reference_docs_text, evidence, reasoning_tree)
        """
        if not getattr(self.retriever, "block_index_enabled", False) or not hasattr(
            self.retriever, "get_gtr_documents_in_blocks"
        ):
            return self.tree_of_thought_without_fusion(query)

        parsed = _extract_log_target(query)
        if not parsed:
            return self.tree_of_thought_without_fusion(query)

        target = parsed["target"]
        is_prefix_query = parsed["is_prefix_query"]

        block_mode = getattr(self.retriever, "block_key_mode", None)
        if not block_mode:
            wiki = getattr(self.args, "wiki_passage", "") or ""
            name = os.path.basename(wiki).lower()
            if "hdfs" in name:
                block_mode = "hdfs"
            elif "bgl" in name:
                block_mode = "bgl"
            elif "thunderbird" in name or "tbird" in name:
                block_mode = "thunderbird"

        if block_mode not in ("hdfs", "bgl", "thunderbird"):
            return self.tree_of_thought_without_fusion(query)

        block_map_enabled = bool(getattr(self.retriever, "block_map_enabled", False))
        max_other_blocks = int(getattr(self.args, "block_chain_other_blocks", 9))
        candidate_pool = int(getattr(self.args, "block_chain_pool", 2000))

        self_block = None
        stripped_target = target
        seed_for_blocks = target

        if block_mode == "hdfs":
            if not block_map_enabled:
                m = _HDFS_BLOCK_RE.search(target) or _HDFS_BLOCK_RE.search(query)
                if m:
                    self_block = m.group(0).lower()
                    stripped_target = _HDFS_BLOCK_RE.sub("", target)
                    stripped_target = " ".join(stripped_target.split()).strip() or target
                    seed_for_blocks = stripped_target
        elif block_mode == "thunderbird":
            if not block_map_enabled:
                host = (target.split(None, 1)[0] if target else "").strip()
                if not host:
                    return self.tree_of_thought_without_fusion(query)
                self_block = host.lower()

                tb_mode = str(getattr(self.args, "block_chain_thunderbird_self_block_mode", "query") or "query").strip().lower()
                if tb_mode == "shift":
                    try:
                        block_to_idx = getattr(self.retriever, "block_to_idx", None) or {}
                        shift_map = getattr(self.retriever, "_block_chain_shift_map", None)
                        if shift_map is None and block_to_idx:
                            keys = sorted(k for k in block_to_idx.keys() if k)
                            shift_map = {keys[i]: keys[(i + 1) % len(keys)] for i in range(len(keys))} if keys else {}
                            setattr(self.retriever, "_block_chain_shift_map", shift_map)
                        if shift_map and self_block in shift_map:
                            self_block = shift_map[self_block]
                    except Exception:
                        pass

        seed_for_global = seed_for_blocks
        if block_mode == "hdfs" and not block_map_enabled:
            block_to_idx = getattr(self.retriever, "block_to_idx", None)
            if not self_block or (block_to_idx is not None and self_block not in block_to_idx):
                seed_for_global = target

        # Fast path: for datasets with a stable self-block key, try self-block retrieval first.
        if block_mode in ("hdfs", "thunderbird") and self_block and not block_map_enabled:
            try:
                seed_self = self.retriever.get_gtr_documents_in_blocks(
                    [seed_for_blocks], [self_block], top_k=self.args.top_k_documents
                )[0]["documents"]
                seed_self_docs = [doc_to_text(d) for d in seed_self]
                fast = self._log_match_fastpath(
                    query, seed_self_docs, mode="block_chain_vote", extra={"self_block": self_block, "other_blocks": []}
                )
                if fast is not None:
                    return fast
            except Exception:
                pass

        global_docs = self.retriever.get_documents(question=[seed_for_global], top_k=candidate_pool)[0]["documents"]

        # Fast path (log-match, prefix queries): lexical scan inside the global pool.
        if int(getattr(self.args, "enable_log_match_fastpath", 1)):
            best_match = None
            for doc in global_docs or []:
                text_norm = " ".join((doc.get("text") or "").split())
                if not text_norm:
                    continue
                if is_prefix_query:
                    if not text_norm.startswith(target):
                        continue
                else:
                    if text_norm != target:
                        continue
                extra = max(0, len(text_norm) - len(target))
                score = float(doc.get("score", 0.0) or 0.0)
                cand = (extra, len(text_norm), -score, text_norm, doc)
                if best_match is None or cand[:4] < best_match[:4]:
                    best_match = cand

            if best_match is not None:
                best_doc = best_match[4]
                head_docs = global_docs[: max(5, self.args.top_k_documents)]
                docs_text = [doc_to_text(best_doc)] + [doc_to_text(d) for d in head_docs if d is not best_doc]
                fast = self._log_match_fastpath(
                    query,
                    docs_text,
                    mode="block_chain_vote",
                    extra={"self_block": self_block, "other_blocks": [], "fastpath_source": "global_pool_lexical"},
                )
                if fast is not None:
                    return fast

        # Fast path: global retrieval already contains an extractive match.
        global_docs_text = [doc_to_text(d) for d in global_docs[: max(5, self.args.top_k_documents)]]
        fast = self._log_match_fastpath(
            query, global_docs_text, mode="block_chain_vote", extra={"self_block": self_block, "other_blocks": []}
        )
        if fast is not None:
            return fast

        def _doc_block_key(doc):
            try:
                doc_id = doc.get("id")
                doc_block_idx = getattr(self.retriever, "doc_block_idx", None)
                block_idx_to_key = getattr(self.retriever, "block_idx_to_key", None)
                if doc_id and doc_block_idx is not None and block_idx_to_key is not None:
                    global_idx = int(doc_id) - 1
                    if global_idx >= 0:
                        blk_idx = doc_block_idx[global_idx]
                        if blk_idx is not None and 0 <= int(blk_idx) < len(block_idx_to_key):
                            key = block_idx_to_key[int(blk_idx)]
                            key_norm = str(key or "").strip().lower()
                            if key_norm and key_norm not in {"__no_block__", "__no_block__"}:
                                return key_norm
            except Exception:
                pass

            if block_mode == "hdfs":
                dm = _HDFS_BLOCK_RE.search(doc.get("text", "")) or _HDFS_BLOCK_RE.search(doc.get("title", ""))
                return dm.group(0).lower() if dm else None
            if block_mode == "bgl":
                return (doc.get("title") or "").strip().lower() or None
            t = (doc.get("text") or "").strip()
            return (t.split(None, 1)[0].strip().lower() if t else None) or None

        def _best_target_match_doc(docs):
            best = None
            for doc in docs or []:
                text_norm = " ".join((doc.get("text") or "").split())
                if not text_norm:
                    continue
                if is_prefix_query:
                    if not text_norm.startswith(target):
                        continue
                else:
                    if text_norm != target:
                        continue
                extra = max(0, len(text_norm) - len(target))
                cand = (extra, len(text_norm), text_norm, doc)
                if best is None or cand[:3] < best[:3]:
                    best = cand
            return best[3] if best else None

        # When super-block mapping is enabled (or for BGL), infer self_block from the global pool.
        if block_mode == "bgl" or (block_map_enabled and block_mode in ("hdfs", "thunderbird")):
            injected_target_doc = _best_target_match_doc(global_docs)
            if injected_target_doc is not None:
                self_block = _doc_block_key(injected_target_doc)
            if not self_block and global_docs:
                self_block = _doc_block_key(global_docs[0])
            if not self_block:
                return self.tree_of_thought_without_fusion(query)

        # Hard ablation: intentionally shift self_block to a wrong block.
        # This makes single-block retrieval fail, so cross-block retrieval can be measured for recovery.
        self_block_original = self_block
        sb_mode = str(getattr(self.args, "block_chain_self_block_mode", "normal") or "normal").strip().lower()
        if (
            sb_mode == "shift"
            and self_block
            and getattr(self.retriever, "block_to_idx", None) is not None
            and getattr(self.retriever, "block_idx_to_key", None) is not None
        ):
            try:
                idx = self.retriever.block_to_idx.get(self_block) if self.retriever.block_to_idx else None
                keys = list(getattr(self.retriever, "block_idx_to_key", None) or [])
                if idx is not None and keys:
                    shifted = keys[(int(idx) + 1) % len(keys)]
                    if shifted and str(shifted).strip().lower() != self_block:
                        self_block = str(shifted).strip().lower()
            except Exception:
                pass

        if not getattr(self.retriever, "block_to_idx", None) or self_block not in self.retriever.block_to_idx:
            return self.tree_of_thought_without_fusion(query)

        seed_self_docs_text = None

        # Fast path (also for inferred self-block under super-block mapping):
        # Try a single self-block retrieval and extractive match before entering ToT DFS.
        # This keeps the framework grounded and avoids expensive LLM calls when the answer
        # exists in the self-block (common for Thunderbird/HDFS log-match).
        if (
            int(getattr(self.args, "enable_log_match_fastpath", 1))
            and block_mode in ("hdfs", "thunderbird")
            and self_block
        ):
            try:
                seed_self = self.retriever.get_gtr_documents_in_blocks(
                    [seed_for_blocks], [self_block], top_k=self.args.top_k_documents
                )[0]["documents"]
                seed_self_docs_text = [doc_to_text(d) for d in seed_self]
                fast = self._log_match_fastpath(
                    query,
                    seed_self_docs_text,
                    mode="block_chain_tot",
                    extra={
                        "self_block": self_block,
                        "other_blocks": [],
                        "fastpath_source": "self_block_retrieval_inferred" if block_map_enabled else "self_block_retrieval",
                    },
                )
                if fast is not None:
                    return fast
            except Exception:
                pass

        best_by_block = {}
        target_program = _extract_thunderbird_program(target) if block_mode == "thunderbird" else ""
        for doc in global_docs:
            bid = _doc_block_key(doc)
            if not bid or bid == self_block:
                continue
            if block_mode == "thunderbird" and target_program:
                dp = _extract_thunderbird_program(doc.get("text", ""))
                if dp and dp != target_program:
                    continue
            prev = best_by_block.get(bid)
            if prev is None or doc.get("score", 0.0) > prev.get("score", 0.0):
                best_by_block[bid] = doc

        other_blocks = [
            bid
            for bid, _ in sorted(best_by_block.items(), key=lambda kv: kv[1].get("score", 0.0), reverse=True)[
                :max_other_blocks
            ]
        ]
        other_blocks = [b for b in other_blocks if b in self.retriever.block_to_idx and b != self_block]

        k_self = getattr(self.args, "block_chain_vote_top_k_self", None)
        k_self = int(k_self) if k_self is not None else int(self.args.top_k_documents)
        if k_self <= 0:
            k_self = int(self.args.top_k_documents)

        k_other = getattr(self.args, "block_chain_vote_top_k_other", None)
        k_other = int(k_other) if k_other is not None else int(getattr(self.args, "block_chain_cross_top_k", 1) or 1)
        if k_other <= 0:
            k_other = int(getattr(self.args, "block_chain_cross_top_k", 1) or 1)
        k_other = max(1, k_other)

        skip_cross_if_self_score_ge = float(getattr(self.args, "block_chain_skip_cross_if_self_score_ge", 1.0))
        cross_program_strict = int(getattr(self.args, "block_chain_cross_program_strict", 0))

        def _best_score(docs):
            if not docs:
                return float("-inf")
            try:
                return max(float(d.get("score", float("-inf")) or float("-inf")) for d in docs)
            except Exception:
                return float("-inf")

        docs_all = []
        candidates = []

        docs_self = self.retriever.get_gtr_documents_in_blocks([seed_for_blocks], [self_block], top_k=k_self)[0][
            "documents"
        ]
        docs_self_text = [doc_to_text(d) for d in (docs_self or [])]

        # Thunderbird/HDFS log-match optimization: the dense top-K inside the block may miss the exact/prefix
        # matching line, while a cheap lexical scan within the block can recover it reliably.
        ans_self = _select_extractive_log_response(query, docs_self_text, response_mode="line")
        if (
            not ans_self
            and int(getattr(self.args, "enable_log_match_fastpath", 1))
            and not block_map_enabled
            and block_mode in ("hdfs", "thunderbird")
            and self_block
            and getattr(self.retriever, "block_to_idx", None)
            and getattr(self.retriever, "block_offsets", None) is not None
            and getattr(self.retriever, "block_doc_indices", None) is not None
        ):
            try:
                blk_idx = self.retriever.block_to_idx.get(self_block) if self.retriever.block_to_idx else None
                if blk_idx is not None:
                    start = self.retriever.block_offsets[blk_idx]
                    end = self.retriever.block_offsets[blk_idx + 1]
                    best = None
                    for global_idx in self.retriever.block_doc_indices[start:end]:
                        title, text = self.retriever.docs[int(global_idx)].split("\n", 1)
                        text_norm = " ".join(text.split())
                        if not text_norm:
                            continue
                        if is_prefix_query:
                            if not text_norm.startswith(target):
                                continue
                            extra = len(text_norm) - len(target)
                            cand = (extra, len(text_norm), int(global_idx), title, text)
                            if best is None or cand[:2] < best[:2]:
                                best = cand
                            if extra == 0:
                                break
                        else:
                            if text_norm != target:
                                continue
                            best = (0, len(text_norm), int(global_idx), title, text)
                            break

                    if best is not None:
                        _, _, global_idx, title, text = best
                        injected = {"id": str(global_idx + 1), "title": title, "text": text, "score": 1.0}
                        injected_text = doc_to_text(injected)
                        if injected_text not in docs_self_text:
                            docs_self_text = [injected_text] + docs_self_text
                        ans_self = _select_extractive_log_response(query, docs_self_text, response_mode="line")
            except Exception:
                pass

        docs_all.extend(docs_self_text)

        ans_self = ans_self or "The information is missing"
        candidates.append(
            {
                "idx": 1,
                "block_id": self_block,
                "answer": (ans_self.splitlines()[0].strip() if ans_self else "The information is missing"),
                "score": _best_score(docs_self),
                "is_self": True,
            }
        )

        self_best_score = _best_score(docs_self)
        do_cross = bool(other_blocks) and (self_best_score < skip_cross_if_self_score_ge)

        if do_cross and other_blocks:
            cross_q = str(seed_for_blocks or "")
            cross_q = _QUERY_PREFIX_RE.sub("", cross_q).strip()
            if cross_q.endswith("..."):
                cross_q = cross_q[:-3].strip()
            if block_mode == "hdfs":
                cross_q = _HDFS_BLOCK_RE.sub("", cross_q)
            cross_q = " ".join(cross_q.split()).strip() or stripped_target

            other_res = self.retriever.get_gtr_documents_in_blocks([cross_q] * len(other_blocks), other_blocks, top_k=k_other)
            for bid, r in zip(other_blocks, other_res):
                docs_blk = list(r.get("documents") or [])

                if block_mode == "thunderbird" and target_program and cross_program_strict and len(docs_blk) > 1:
                    filtered = []
                    for d in docs_blk:
                        dp = _extract_thunderbird_program(d.get("text", ""))
                        if dp and dp == target_program:
                            filtered.append(d)
                    docs_blk = filtered

                docs_blk_text = [doc_to_text(d) for d in docs_blk]
                docs_all.extend(docs_blk_text)
                ans = _select_extractive_log_response(query, docs_blk_text, response_mode="line") or "The information is missing"
                candidates.append(
                    {
                        "idx": len(candidates) + 1,
                        "block_id": bid,
                        "answer": (ans.splitlines()[0].strip() if ans else "The information is missing"),
                        "score": _best_score(docs_blk),
                        "is_self": False,
                    }
                )

        # Dedupe documents while preserving order (keeps grounding small and stable).
        seen = set()
        docs_text = []
        for t in docs_all:
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            docs_text.append(t)

        def _parse_winner(raw: str, n: int):
            if not raw:
                return None
            m = re.search(r"(?i)winner\\s*[:=]\\s*(\\d+)", str(raw))
            if not m:
                m = re.search(r"\\b(\\d+)\\b", str(raw).strip())
            if not m:
                return None
            try:
                idx = int(m.group(1))
            except Exception:
                return None
            if idx < 1 or idx > n:
                return None
            return idx - 1

        vote_prompt = None
        vote_raw = None
        winner_i = 0

        if len(candidates) > 1:
            try:
                vote_prompt = self.vote_generator.get_vote_prompt(question=query, candidates=candidates)
                vote_raw = self._generate_with_model(self.vote_generator, "", vote_prompt)
            except Exception:
                vote_raw = ""
            parsed_winner = _parse_winner(vote_raw, len(candidates))
            if parsed_winner is not None:
                winner_i = parsed_winner
            else:
                # Fallback: pick the best-scoring non-missing candidate.
                best = None
                for i, c in enumerate(candidates):
                    ans = str(c.get("answer") or "")
                    if not ans or "information is missing" in ans.lower():
                        continue
                    s = float(c.get("score", float("-inf")) or float("-inf"))
                    cand = (s, int(bool(c.get("is_self"))), -i)
                    if best is None or cand > best[0]:
                        best = (cand, i)
                if best is not None:
                    winner_i = best[1]

        response = str(candidates[winner_i].get("answer") or "").strip() if candidates else "The information is missing"
        response = _maybe_truncate_to_target_prefix(self.args, query, docs_text, response)
        if response:
            response = response.splitlines()[0].strip()

        reasoning = {
            "question": query,
            "mode": "block_chain_vote",
            "self_block": self_block,
            "other_blocks": other_blocks,
            "candidates": candidates,
            "winner_idx": int(winner_i) + 1 if candidates else None,
            "winner": candidates[winner_i] if candidates else None,
            "vote_prompt": vote_prompt,
            "vote_raw": vote_raw,
        }

        return response, docs_text, [], reasoning
    def tree_of_thought_with_cross_block(self, query):
        if not getattr(self.retriever, "block_index_enabled", False) or not hasattr(
            self.retriever, "get_gtr_documents_in_blocks"
        ):
            return self.tree_of_thought_without_fusion(query)

        parsed = _extract_log_target(query)
        if not parsed:
            return self.tree_of_thought_without_fusion(query)

        target = parsed["target"]
        is_prefix_query = parsed["is_prefix_query"]

        block_mode = getattr(self.retriever, "block_key_mode", None)
        if not block_mode:
            wiki = getattr(self.args, "wiki_passage", "") or ""
            name = os.path.basename(wiki).lower()
            if "hdfs" in name:
                block_mode = "hdfs"
            elif "bgl" in name:
                block_mode = "bgl"
            elif "thunderbird" in name or "tbird" in name:
                block_mode = "thunderbird"

        if block_mode not in ("hdfs", "bgl", "thunderbird"):
            return self.tree_of_thought_without_fusion(query)

        block_map_enabled = bool(getattr(self.retriever, "block_map_enabled", False))

        max_other_blocks = int(getattr(self.args, "block_chain_other_blocks", 9))
        candidate_pool = int(getattr(self.args, "block_chain_pool", 2000))
        min_similarity = float(getattr(self.args, "block_chain_min_similarity", 0.35))

        injected_target_doc = None
        stripped_target = target

        if block_mode == "hdfs":
            if block_map_enabled:
                # When super-block mapping is enabled, self_block cannot be derived from the raw query key.
                # We infer it later from the global retrieval pool (BGL-style).
                self_block = None
                seed_for_blocks = target
            else:
                m = _HDFS_BLOCK_RE.search(target) or _HDFS_BLOCK_RE.search(query)
                if m:
                    self_block = m.group(0).lower()
                    stripped_target = _HDFS_BLOCK_RE.sub("", target)
                    stripped_target = " ".join(stripped_target.split()).strip() or target
                    seed_for_blocks = stripped_target
                else:
                    # Some HDFS log-match queries are truncated before the full block id appears (or omit it entirely).
                    # We still attempt a global pool + lexical prefix match before falling back to standard ToT.
                    self_block = None
                    seed_for_blocks = target
        elif block_mode == "thunderbird":
            if block_map_enabled:
                # When super-block mapping is enabled, self_block cannot be derived from the raw host key.
                # We infer it later from the global retrieval pool (BGL-style).
                self_block = None
                seed_for_blocks = target
            else:
                host = (target.split(None, 1)[0] if target else "").strip()
                if not host:
                    return self.tree_of_thought_without_fusion(query)
                self_block = host.lower()

                tb_mode = str(getattr(self.args, "block_chain_thunderbird_self_block_mode", "query") or "query").strip().lower()
                if tb_mode == "shift":
                    try:
                        block_to_idx = getattr(self.retriever, "block_to_idx", None) or {}
                        shift_map = getattr(self.retriever, "_block_chain_shift_map", None)
                        if shift_map is None and block_to_idx:
                            keys = sorted(k for k in block_to_idx.keys() if k)
                            shift_map = {keys[i]: keys[(i + 1) % len(keys)] for i in range(len(keys))} if keys else {}
                            setattr(self.retriever, "_block_chain_shift_map", shift_map)
                        if shift_map and self_block in shift_map:
                            self_block = shift_map[self_block]
                    except Exception:
                        pass
                seed_for_blocks = target
        else:
            # BGL: treat title as block key and infer the best title from a global retrieval pool.
            self_block = None
            seed_for_blocks = target

        seed_for_global = seed_for_blocks
        if block_mode == "hdfs":
            # If we can't rely on the self-block index key (truncated/missing blk_...), include TARGET itself
            # so the candidate pool is more likely to contain the exact/prefix-matching line.
            block_to_idx = getattr(self.retriever, "block_to_idx", None)
            if not self_block or (block_to_idx is not None and self_block not in block_to_idx):
                seed_for_global = target

            # HDFS-specific: some queries contain a truncated block id (blk_... ends right before "..."),
            # so the extracted self_block key does not exist in block_to_idx. Dense global retrieval often
            # misses the exact/prefix-matching line in this case; instead, resolve candidate blocks by
            # prefix and scan within those blocks for a lexical startswith(TARGET) match.
            if (
                int(getattr(self.args, "enable_log_match_fastpath", 1))
                and int(getattr(self.args, "hdfs_block_prefix_scan", 1))
                and not block_map_enabled
                and self_block
                and is_prefix_query
                and block_to_idx is not None
                and self_block not in block_to_idx
                and getattr(self.retriever, "block_offsets", None) is not None
                and getattr(self.retriever, "block_doc_indices", None) is not None
            ):
                try:
                    import bisect

                    keys_sorted = getattr(self.retriever, "_block_keys_sorted", None)
                    if keys_sorted is None:
                        keys_sorted = sorted(block_to_idx.keys())
                        setattr(self.retriever, "_block_keys_sorted", keys_sorted)

                    lo = bisect.bisect_left(keys_sorted, self_block)
                    hi = bisect.bisect_right(keys_sorted, self_block + "\uffff")
                    candidates = keys_sorted[lo:hi]

                    max_candidates = int(getattr(self.args, "block_chain_prefix_block_candidates", 50))
                    if max_candidates > 0 and len(candidates) > max_candidates:
                        candidates = candidates[:max_candidates]

                    best = None
                    for bid in candidates:
                        blk_idx = block_to_idx.get(bid)
                        if blk_idx is None:
                            continue
                        start = self.retriever.block_offsets[blk_idx]
                        end = self.retriever.block_offsets[blk_idx + 1]
                        for global_idx in self.retriever.block_doc_indices[start:end]:
                            title, text = self.retriever.docs[int(global_idx)].split("\n", 1)
                            text_norm = " ".join(text.split())
                            if not text_norm.startswith(target):
                                continue
                            extra = len(text_norm) - len(target)
                            cand = (extra, len(text_norm), int(global_idx), bid, title, text)
                            if best is None or cand[:2] < best[:2]:
                                best = cand
                            if extra == 0:
                                break
                        if best is not None and best[0] == 0:
                            break

                    if best is not None:
                        _, _, global_idx, resolved_block, title, text = best
                        doc = {"id": str(global_idx + 1), "title": title, "text": text, "score": 1.0}
                        fast = self._log_match_fastpath(
                            query,
                            [doc_to_text(doc)],
                            mode="block_chain_tot",
                            extra={
                                "self_block": resolved_block,
                                "other_blocks": [],
                                "fastpath_source": "hdfs_block_prefix_scan",
                            },
                        )
                        if fast is not None:
                            return fast
                except Exception:
                    pass

        # Thunderbird optimization for log-match prefix queries:
        # Directly scan within the host block for a line starting with TARGET. This is a cheap lexical
        # check inside the already-built block index and avoids expensive ToT/LLM calls when the exact
        # prefix exists in the corpus (which is common for this dataset format).
        if (
            int(getattr(self.args, "enable_log_match_fastpath", 1))
            and block_mode == "thunderbird"
            and not block_map_enabled
            and self_block
            and is_prefix_query
            and getattr(self.retriever, "block_to_idx", None)
        ):
            try:
                blk_idx = self.retriever.block_to_idx.get(self_block) if self.retriever.block_to_idx else None
                if blk_idx is not None and self.retriever.block_offsets is not None and self.retriever.block_doc_indices is not None:
                    start = self.retriever.block_offsets[blk_idx]
                    end = self.retriever.block_offsets[blk_idx + 1]
                    best = None
                    for global_idx in self.retriever.block_doc_indices[start:end]:
                        title, text = self.retriever.docs[int(global_idx)].split("\n", 1)
                        text_norm = " ".join(text.split())
                        if not text_norm.startswith(target):
                            continue
                        extra = len(text_norm) - len(target)
                        cand = (extra, len(text_norm), int(global_idx), title, text)
                        if best is None or cand[:2] < best[:2]:
                            best = cand
                        if extra == 0:
                            break
                    if best is not None:
                        doc = {"id": str(best[2] + 1), "title": best[3], "text": best[4], "score": 1.0}
                        fast = self._log_match_fastpath(
                            query,
                            [doc_to_text(doc)],
                            mode="block_chain_tot",
                            extra={"self_block": self_block, "other_blocks": [], "fastpath_source": "self_block_prefix_scan"},
                        )
                        if fast is not None:
                            return fast
            except Exception:
                pass

        # Fast path: for datasets with a stable self-block key, try self-block retrieval first.
        # If it already contains an extractive match, skip ToT entirely.
        if block_mode in ("hdfs", "thunderbird") and self_block:
            try:
                seed_self = self.retriever.get_gtr_documents_in_blocks(
                    [seed_for_blocks], [self_block], top_k=self.args.top_k_documents
                )[0]["documents"]
                seed_self_docs = [doc_to_text(d) for d in seed_self]
                fast = self._log_match_fastpath(
                    query, seed_self_docs, mode="block_chain_tot", extra={"self_block": self_block, "other_blocks": []}
                )
                if fast is not None:
                    return fast
            except Exception:
                pass

        global_docs = self.retriever.get_documents(question=[seed_for_global], top_k=candidate_pool)[0]["documents"]

        # Fast path (log-match, prefix queries): search the whole candidate pool for an exact/prefix match.
        # This is a lightweight lexical rerank on the already-retrieved pool and dramatically reduces
        # LLM calls for datasets like Thunderbird where the query is a syslog prefix ending with "...".
        if int(getattr(self.args, "enable_log_match_fastpath", 1)):
            best_match = None
            for doc in global_docs or []:
                text_norm = " ".join((doc.get("text") or "").split())
                if not text_norm:
                    continue
                if is_prefix_query:
                    if not text_norm.startswith(target):
                        continue
                else:
                    if text_norm != target:
                        continue
                extra = max(0, len(text_norm) - len(target))
                score = float(doc.get("score", 0.0) or 0.0)
                cand = (extra, len(text_norm), -score, text_norm, doc)
                if best_match is None or cand[:4] < best_match[:4]:
                    best_match = cand

            if best_match is not None:
                best_doc = best_match[4]
                head_docs = global_docs[: max(5, self.args.top_k_documents)]
                docs_text = [doc_to_text(best_doc)] + [doc_to_text(d) for d in head_docs if d is not best_doc]
                fast = self._log_match_fastpath(
                    query,
                    docs_text,
                    mode="block_chain_tot",
                    extra={"self_block": self_block, "other_blocks": [], "fastpath_source": "global_pool_lexical"},
                )
                if fast is not None:
                    return fast

        # Fast path: global retrieval already contains an extractive match.
        global_docs_text = [doc_to_text(d) for d in global_docs[: max(5, self.args.top_k_documents)]]
        fast = self._log_match_fastpath(
            query, global_docs_text, mode="block_chain_tot", extra={"self_block": self_block, "other_blocks": []}
        )
        if fast is not None:
            return fast

        def _doc_block_key(doc):
            # Prefer the retriever's built block index mapping when available.
            # This is required for super-block ablations where block keys are not directly derivable from doc text.
            try:
                doc_id = doc.get("id")
                doc_block_idx = getattr(self.retriever, "doc_block_idx", None)
                block_idx_to_key = getattr(self.retriever, "block_idx_to_key", None)
                if doc_id and doc_block_idx is not None and block_idx_to_key is not None:
                    global_idx = int(doc_id) - 1
                    if global_idx >= 0:
                        blk_idx = doc_block_idx[global_idx]
                        if blk_idx is not None and 0 <= int(blk_idx) < len(block_idx_to_key):
                            key = block_idx_to_key[int(blk_idx)]
                            key_norm = str(key or "").strip().lower()
                            if key_norm and key_norm not in {"__no_block__", "__no_block__"}:
                                return key_norm
            except Exception:
                pass

            # Fallback: infer from document fields.
            if block_mode == "hdfs":
                dm = _HDFS_BLOCK_RE.search(doc.get("text", "")) or _HDFS_BLOCK_RE.search(doc.get("title", ""))
                return dm.group(0).lower() if dm else None
            if block_mode == "bgl":
                return (doc.get("title") or "").strip().lower() or None
            t = (doc.get("text") or "").strip()
            return (t.split(None, 1)[0].strip().lower() if t else None) or None

        def _best_target_match_doc(docs):
            best = None
            for doc in docs or []:
                text_norm = " ".join((doc.get("text") or "").split())
                if not text_norm:
                    continue
                if is_prefix_query:
                    if not text_norm.startswith(target):
                        continue
                else:
                    if text_norm != target:
                        continue
                extra = max(0, len(text_norm) - len(target))
                cand = (extra, len(text_norm), text_norm, doc)
                if best is None or cand[:3] < best[:3]:
                    best = cand
            return best[3] if best else None

        if block_mode == "bgl" or (block_map_enabled and block_mode in ("hdfs", "thunderbird")):
            injected_target_doc = _best_target_match_doc(global_docs)
            if injected_target_doc is not None:
                self_block = _doc_block_key(injected_target_doc)
            if not self_block and global_docs:
                self_block = _doc_block_key(global_docs[0])
            if not self_block:
                return self.tree_of_thought_without_fusion(query)
        else:
            injected_candidate = _best_target_match_doc(global_docs)
            if injected_candidate is not None and _doc_block_key(injected_candidate) == self_block:
                injected_target_doc = injected_candidate

        # Hard ablation: intentionally shift self_block to a wrong block.
        # This makes single-block retrieval fail, so cross-block retrieval can be measured for recovery.
        self_block_original = self_block
        sb_mode = str(getattr(self.args, "block_chain_self_block_mode", "normal") or "normal").strip().lower()
        if (
            sb_mode == "shift"
            and self_block
            and getattr(self.retriever, "block_to_idx", None) is not None
            and getattr(self.retriever, "block_idx_to_key", None) is not None
        ):
            try:
                idx = self.retriever.block_to_idx.get(self_block) if self.retriever.block_to_idx else None
                keys = list(getattr(self.retriever, "block_idx_to_key", None) or [])
                if idx is not None and keys:
                    shifted = keys[(int(idx) + 1) % len(keys)]
                    if shifted and str(shifted).strip().lower() != self_block:
                        self_block = str(shifted).strip().lower()
            except Exception:
                pass

        if not getattr(self.retriever, "block_to_idx", None) or self_block not in self.retriever.block_to_idx:
            return self.tree_of_thought_without_fusion(query)

        best_by_block = {}
        target_program = _extract_thunderbird_program(target) if block_mode == "thunderbird" else ""
        for doc in global_docs:
            bid = _doc_block_key(doc)
            if not bid or bid == self_block:
                continue
            if block_mode == "thunderbird" and target_program:
                dp = _extract_thunderbird_program(doc.get("text", ""))
                if dp and dp != target_program:
                    continue
            prev = best_by_block.get(bid)
            if prev is None or doc.get("score", 0.0) > prev.get("score", 0.0):
                best_by_block[bid] = doc

        other_blocks = [
            bid
            for bid, _ in sorted(best_by_block.items(), key=lambda kv: kv[1].get("score", 0.0), reverse=True)[
                :max_other_blocks
            ]
        ]
        other_blocks = [b for b in other_blocks if b in self.retriever.block_to_idx and b != self_block]

        def _maybe_inject_target_match(documents):
            if not is_prefix_query:
                return documents

            for d in documents:
                if " ".join((d.get("text") or "").split()).startswith(target):
                    return documents

            if injected_target_doc is not None and _doc_block_key(injected_target_doc) == self_block:
                injected_id = injected_target_doc.get("id")
                if injected_id and injected_id not in {d.get("id") for d in documents}:
                    return [injected_target_doc] + list(documents)

            # Avoid full scans for BGL titles (can be extremely large).
            if block_mode not in ("hdfs", "thunderbird"):
                return documents
            if block_map_enabled:
                return documents

            blk_idx = self.retriever.block_to_idx.get(self_block) if self.retriever.block_to_idx else None
            if blk_idx is None or self.retriever.block_offsets is None or self.retriever.block_doc_indices is None:
                return documents

            start = self.retriever.block_offsets[blk_idx]
            end = self.retriever.block_offsets[blk_idx + 1]
            best = None
            for global_idx in self.retriever.block_doc_indices[start:end]:
                title, text = self.retriever.docs[int(global_idx)].split("\n", 1)
                text_norm = " ".join(text.split())
                if not text_norm.startswith(target):
                    continue
                extra = len(text_norm) - len(target)
                cand = (extra, len(text_norm), int(global_idx), title, text)
                if best is None or cand[:2] < best[:2]:
                    best = cand
                if extra == 0:
                    break

            if best is None:
                return documents

            _, _, global_idx, title, text = best
            injected = {"id": str(global_idx + 1), "title": title, "text": text, "score": 1.0}
            ids = {d.get("id") for d in documents}
            if injected["id"] in ids:
                return documents
            return [injected] + list(documents)

        seed_self_docs_text = None

        # When super-block mapping is enabled, the self-block is inferred from a global pool.
        # Before entering the expensive ToT DFS, do a single self-block retrieval and attempt the
        # same grounded extractive fastpath as the native (host/blk_) block setting.
        if (
            block_map_enabled
            and int(getattr(self.args, "enable_log_match_fastpath", 1))
            and block_mode in ("hdfs", "thunderbird")
            and self_block
        ):
            try:
                seed_self = self.retriever.get_gtr_documents_in_blocks(
                    [seed_for_blocks], [self_block], top_k=self.args.top_k_documents
                )[0]["documents"]
                seed_self = _maybe_inject_target_match(seed_self)
                seed_self_docs_text = [doc_to_text(d) for d in seed_self]
                fast = self._log_match_fastpath(
                    query,
                    seed_self_docs_text,
                    mode="block_chain_tot",
                    extra={
                        "self_block": self_block,
                        "other_blocks": other_blocks,
                        "fastpath_source": "self_block_retrieval_inferred",
                    },
                )
                if fast is not None:
                    return fast
            except Exception:
                pass

        def _retrieve_documents(question_text, depth):
            if self.args.max_nodes is not None and depth <= len(self.args.max_nodes):
                k_self = self.args.max_nodes[depth - 1]
            else:
                k_self = self.args.top_k_documents

            docs_self = self.retriever.get_gtr_documents_in_blocks([question_text], [self_block], top_k=k_self)[0][
                "documents"
            ]
            if depth == 1:
                docs_self = _maybe_inject_target_match(docs_self)
                nonlocal seed_self_docs_text
                if seed_self_docs_text is None:
                    seed_self_docs_text = [doc_to_text(d) for d in docs_self]

            docs = list(docs_self)

            cross_top_k = int(getattr(self.args, "block_chain_cross_top_k", 1))
            skip_cross_if_self_score_ge = float(getattr(self.args, "block_chain_skip_cross_if_self_score_ge", 1.0))
            cross_program_strict = int(getattr(self.args, "block_chain_cross_program_strict", 0))

            self_best_score = 0.0
            if docs_self:
                try:
                    self_best_score = max(float(d.get("score", 0.0) or 0.0) for d in docs_self)
                except Exception:
                    self_best_score = 0.0

            if other_blocks and self_best_score < skip_cross_if_self_score_ge:
                cross_q = str(question_text or "")
                cross_q = _QUERY_PREFIX_RE.sub("", cross_q).strip()
                if cross_q.endswith("..."):
                    cross_q = cross_q[:-3].strip()
                if block_mode == "hdfs":
                    cross_q = _HDFS_BLOCK_RE.sub("", cross_q)
                cross_q = " ".join(cross_q.split()).strip() or stripped_target

                other_res = self.retriever.get_gtr_documents_in_blocks(
                    [cross_q] * len(other_blocks), other_blocks, top_k=max(1, cross_top_k)
                )
                cross_candidates = []
                for r in other_res:
                    candidates = list(r.get("documents") or [])
                    if not candidates:
                        continue

                    if block_mode == "thunderbird" and target_program and cross_program_strict:
                        filtered = []
                        for d in candidates:
                            dp = _extract_thunderbird_program(d.get("text", ""))
                            if dp and dp == target_program:
                                filtered.append(d)
                        candidates = filtered
                        if not candidates:
                            continue

                    best = None
                    best_score = float("-inf")
                    for d in candidates:
                        s = float(d.get("score", float("-inf")))
                        if s > best_score:
                            best_score = s
                            best = d
                    if best is not None:
                        cross_candidates.append(best)

                if cross_candidates:
                    append_mode = str(getattr(self.args, "block_chain_cross_append_mode", "best") or "best").strip().lower()
                    if append_mode == "all":
                        ids = {d.get("id") for d in docs}
                        for cand in cross_candidates:
                            if float(cand.get("score", 0.0) or 0.0) < min_similarity:
                                continue
                            cid = cand.get("id")
                            if cid and cid in ids:
                                continue
                            docs.append(cand)
                            if cid:
                                ids.add(cid)
                    else:
                        best_cross = max(cross_candidates, key=lambda d: d.get("score", float("-inf")))
                        if float(best_cross.get("score", 0.0)) >= min_similarity:
                            if best_cross.get("id") not in {d.get("id") for d in docs}:
                                docs.append(best_cross)

            return docs

        if int(getattr(self.args, "log_match_skip_thought", 0)):
            try:
                seed_docs = _retrieve_documents(query, depth=1)
                seed_docs_text = [doc_to_text(d) for d in seed_docs]
                response = _select_extractive_log_response(
                    query, seed_docs_text, response_mode=getattr(self.args, "log_response_mode", "line")
                )
                if response is None:
                    response = "The information is missing"
                response = _maybe_truncate_to_target_prefix(self.args, query, seed_docs_text, response)
                if response:
                    response = response.splitlines()[0].strip()
                reasoning = {
                    "question": query,
                    "mode": "block_chain_tot",
                    "self_block": self_block,
                    "other_blocks": other_blocks,
                    "fastpath": False,
                    "skip_thought": True,
                    "nodes": [],
                }
                return response, seed_docs_text, [], reasoning
            except Exception:
                pass

        # Fast path (super-block): run the depth-1 block retrieval once and try an extractive match.
        # If it succeeds, skip the ToT DFS and avoid LLM calls entirely.
        if block_map_enabled and int(getattr(self.args, "enable_log_match_fastpath", 1)):
            try:
                seed_docs = _retrieve_documents(seed_for_blocks, depth=1)
                seed_docs_text = [doc_to_text(d) for d in seed_docs]
                fast = self._log_match_fastpath(
                    query,
                    seed_docs_text,
                    mode="block_chain_tot",
                    extra={
                        "self_block": self_block,
                        "other_blocks": other_blocks,
                        "fastpath_source": "block_retrieval_depth1",
                    },
                )
                if fast is not None:
                    return fast
            except Exception:
                pass

        def dfs(question_text, history, supported, depth):
            layer_nodes = []
            logger.info("Search Depth:{}".format(depth))
            if depth > self.args.max_depth:
                return history, supported, layer_nodes

            documents = _retrieve_documents(question_text, depth)
            for cnt, document in enumerate(documents):
                logger.info("Thought of the {} node in depth {}.".format(cnt + 1, depth))
                if not self.check_document_exist(supported=supported, document=document):
                    logger.warning("document exist in supported evidence path, skip and continue")
                    continue

                text = doc_to_text(document)
                history.append({"id": document["id"], "text": text})
                history_docs = [his["text"] for his in history]
                thought_prompt = self.generator.get_thought_prompt(self.args.thought_shot, history_docs, query)
                response = self.generate("", thought_prompt)
                if "gpt" in self.args.generator and response == "":
                    response = self.retry_loop("", thought_prompt)

                response_labels = self.parsing_thought(response)
                leaf = {
                    "depth": depth,
                    "nid": cnt,
                    "inputs": thought_prompt,
                    "outputs": response,
                    "labels": response_labels,
                    "state": "Failed" if response_labels is None else "Success",
                    "child": [],
                }
                layer_nodes.append(leaf)

                if response_labels is None:
                    self.failed_thought += 1
                    if self.args.failed_parse_file is not None:
                        with open(self.args.failed_parse_file, "a") as f:
                            tmp = {"error_type": "Thought_prompt", "prompt": thought_prompt, "response": response}
                            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
                    history.pop(-1)
                    continue

                self.success_thought += 1
                if response_labels["decision"] == "reject":
                    history.pop(-1)
                elif response_labels["decision"] == "continue":
                    new_question = self.extract_query_content(response_labels["answer_content"], fallback_query=question_text)
                    history, supported, child_nodes = dfs(new_question, history, supported, depth + 1)
                    leaf["child"] = child_nodes
                    history.pop(-1)
                else:
                    supported.append({"evidence": response_labels["answer_content"], "history_docs": copy.deepcopy(history)})
                    history.pop(-1)

            return history, supported, layer_nodes

        history, supported, nodes = dfs(query, [], [], 1)

        evidence_tree = {
            "question": query,
            "mode": "block_chain_tot",
            "self_block": self_block,
            "other_blocks": other_blocks,
            "nodes": nodes,
        }

        docs_idx = []
        docs = []
        for supporting in supported:
            for document in supporting["history_docs"]:
                if document["id"] not in docs_idx:
                    docs_idx.append(document["id"])
                    docs.append(document["text"])

        # For log-matching tasks, always include the initial self-block retrieval docs as answer candidates.
        # This prevents empty-reference cases when ToT rejects every node and improves grounding.
        if seed_self_docs_text:
            seen = set(docs)
            for text in seed_self_docs_text:
                if text not in seen:
                    seen.add(text)
                    docs.append(text)

        response = _select_extractive_log_response(query, docs, response_mode=getattr(self.args, "log_response_mode", "line"))
        if not int(getattr(self.args, "use_extractive_response", 1)):
            response = None
        if response is None:
            response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, docs)
            response = self.generate("", response_prompt)
            if "gpt" in self.args.generator and response == "":
                response = self.retry_loop("", response_prompt)

        # Optional: block+ToT can post-process grounded full-line copies to TARGET prefix for prefix queries.
        response = _maybe_truncate_to_target_prefix(self.args, query, docs, response)
        if response:
            response = response.splitlines()[0].strip()

        return response, docs, supported, evidence_tree
