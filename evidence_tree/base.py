import logging
import copy
import os
import re

import dist_utils
import torch.distributed as dist

from evidence_tree.utils import (
    _select_extractive_log_response,
    _maybe_truncate_to_target_prefix,
)

logger = logging.getLogger(__name__)


class TreeOfEvidenceBase:
    def __init__(self, retriever, generator, args):
        self.retriever = retriever
        self.generator = generator
        self.vote_generator = generator
        self.success_thought = 0
        self.failed_thought = 0
        self.args = args
        # Pruning statistics
        self.consecutive_rejects = 0
        self.max_consecutive_rejects = getattr(args, 'max_consecutive_rejects', 3)
        self.total_queries_generated = 0

        judge_model = str(getattr(args, "block_chain_vote_judge_model", "") or "").strip()
        if judge_model and judge_model != getattr(args, "generator", None):
            try:
                import copy as _copy
                from generator import initial_generator

                judge_args = _copy.copy(args)
                judge_args.generator = judge_model
                self.vote_generator = initial_generator(judge_args)
            except Exception:
                self.vote_generator = generator

    def _log_match_fastpath(self, query, docs, mode, extra=None):
        """
        Optimization for log-match experiments:
        If an extractive match is already present in the currently retrieved docs, return immediately
        (skip ToT/LLM). This keeps the framework grounded while making large-sample runs tractable.
        """
        if not int(getattr(self.args, "enable_log_match_fastpath", 1)):
            return None
        resp = _select_extractive_log_response(query, docs, response_mode=getattr(self.args, "log_response_mode", "line"))
        if not resp:
            return None

        # Optional: unify prefix truncation across modes (tot / block_chain_tot) under the same flag.
        resp = _maybe_truncate_to_target_prefix(self.args, query, docs, resp)

        meta = {"question": query, "mode": mode, "fastpath": True, "nodes": []}
        if extra:
            meta.update(extra)
        return resp, docs, [], meta

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
    def check_document_exist(self, supported, document):
        for supporting in supported:
            for sup_doc in supporting["history_docs"]:
                if document["id"] == sup_doc["id"]:
                    return False
        return True
    def logger_rank0(self, info, level="warning"):
        if not dist.is_initialized() or dist_utils.get_rank() == 0:
            if level == "warning":
                logger.warning(info)
            else:
                logger.info(info)
        return
    def extract_query_content(self, raw_query, fallback_query=None):
        """
        Extract clean query content from LLM output.
        Handles formats like: [QUERY] Find subsequent ERROR logs for Block blk_xxx
        """
        query = (raw_query or "").strip()
        # Remove residual [QUERY] tag
        query = re.sub(r"^\[query\]\s*", "", query, flags=re.IGNORECASE).strip()
        # Some models concatenate next section title after [QUERY]; only take first non-empty line.
        if query:
            for line in query.splitlines():
                line = line.strip()
                if line:
                    query = line
                    break
            else:
                query = ""
        # Remove extra quotes / Markdown prefixes
        query = query.strip("\"'")
        query = re.sub(r"^[#>*\\-\\s]+", "", query).strip()
        query = " ".join(query.split()).strip()
        # If empty, return original question as fallback
        if not query:
            logger.warning("Empty query extracted, using fallback query")
            return (fallback_query or raw_query).strip()
        return query
    def retry_loop(self, system_prompt, inputs):
        response = ""
        repeats = 0
        while response == "" and repeats <= self.generator.repeat_times:
            response = self.generate(system_prompt, inputs)
            repeats += 1
        return response

    def generate(self, system_prompt, inputs):
        if "gpt" in self.args.generator:
            body = self.generator.get_body(system_prompt=system_prompt, input=inputs)
            response = self.generator.get_response(body=body)
        else:
            response = self.generator.get_response(queries=[inputs], system_prompt=system_prompt)
        return response
