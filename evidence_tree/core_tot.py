import json
import logging
import copy
import re
import os

import dist_utils
import torch.distributed as dist

from evidence_tree.utils import (
    doc_to_text,
    _select_extractive_log_response,
    _maybe_truncate_to_target_prefix,
)

logger = logging.getLogger(__name__)


class CoreToTMixin:
    def tree_of_thought(self, query):

        def dfs(question, history, supported, depth):
            if depth > self.args.max_depth:
                return history, supported
            if self.args.max_nodes is not None and depth <= len(self.args.max_nodes):
                response = self.retriever.get_documents(question=[question], top_k=self.args.max_nodes[depth-1])
            else:
                response = self.retriever.get_documents(question=[question], top_k=self.args.top_k_documents)
            for document in response["documents"]:
                text = doc_to_text(document)
                history.append({
                    "id": document["id"],
                    "text": text,
                })
                system_prompt, thought_prompt = self.generator.get_thought_prompt(self.args.thought_shot, history)
                response = self.generator(system_prompt, thought_prompt)
                choice = response.strip().split("\n")[-1]
                choice = choice.lower()

                if "accept" in choice:
                    supported = self.evidence_fusion(supported, history, question)
                    history.pop(-1)
                    return history, supported
                elif "continue" in choice:
                    system_prompt, query_prompt = self.generator.get_missing_evidence_prompt(self.args.missing_evidence_shot, history)
                    new_query = self.generate(system_prompt, query_prompt)
                    history, supported = dfs(new_query, history, supported)
                    history.pop(-1)
                    return history, supported
                else:
                    history.pop(-1)
                    return history, supported

        history, supported = dfs(query, [], [], 1)

        docs_idx = []
        docs = []
        for supporting in supported:
            for document in supporting["history"]:
                if document["id"] not in docs_idx:
                    docs_idx.append(document["id"])
                    docs.append(document["text"])

        system_prompt, response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, docs)
        response = self.generate(system_prompt, response_prompt)
        return response
    def tree_of_thought_without_fusion(self, query):
        # Seed retrieval (used for fastpath and for optional retrieval-only log-match ablations).
        aiops_skip = int(getattr(self.args, "aiops_skip_thought", 0) or 0)
        # For AIOps anomaly-detection queries, thought parsing expects Step 1..4 and may need a longer generation budget.
        kind = str(getattr(self.args, "aiops_query_kind", "") or "").strip().lower()
        is_hdfs_seq = bool(re.search(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+", query or ""))
        is_message = bool(re.search(r"(?i)^\s*(?:\\[query\\]\\s*)?find\s+logs\s+with\s+message\s*:", query or ""))
        if kind == "hdfs_seq" and not is_hdfs_seq:
            is_hdfs_seq = False
        if kind == "message" and not is_message:
            is_message = False
        is_aiops_query = bool(is_hdfs_seq or is_message)
        thought_max_len = int(getattr(self.args, "max_gen_len", 256) or 256)
        if is_aiops_query:
            thought_max_len = int(getattr(self.args, "aiops_thought_max_gen_len", thought_max_len) or thought_max_len)
        seed_docs = []
        try:
            seed = self.retriever.get_documents(question=[query], top_k=self.args.top_k_documents)[0]["documents"]
            seed_docs = [doc_to_text(d) for d in seed]
            # NOTE: AIOps anomaly detection queries share the same "Find logs with message: ..." format,
            # but the desired output is a Normal/Anomaly judgment, not a matched log line. When the user
            # enables --aiops_skip_thought, bypass log-match fastpaths so we don't short-circuit the task.
            if not aiops_skip:
                fast = self._log_match_fastpath(query, seed_docs, mode="tot")
                if fast is not None:
                    return fast
        except Exception:
            seed_docs = []

        if (not aiops_skip) and int(getattr(self.args, "log_match_skip_thought", 0)) and _extract_log_target(query):
            response = _select_extractive_log_response(
                query, seed_docs, response_mode=getattr(self.args, "log_response_mode", "line")
            )
            if response is None:
                response = "The information is missing"
            response = _maybe_truncate_to_target_prefix(self.args, query, seed_docs, response)
            if response:
                response = response.splitlines()[0].strip()
            reasoning = {"question": query, "mode": "tot", "fastpath": False, "skip_thought": True, "nodes": []}
            return response, seed_docs, [], reasoning

        # AIOps ablation: for HDFS sequence anomaly queries, optionally skip ToT DFS and decide via
        # a single global retrieval + response prompt (isolates retrieval effects).
        if int(getattr(self.args, "aiops_skip_thought", 0) or 0):
            try:
                kind = str(getattr(self.args, "aiops_query_kind", "hdfs_seq") or "hdfs_seq").strip().lower()
                is_hdfs_seq = bool(re.search(r"(?i)analyze\s+log\s+sequence\s+for\s+block\s+blk_-?\d+", query or ""))
                is_message = bool(re.search(r"(?i)^\s*(?:\\[query\\]\\s*)?find\s+logs\s+with\s+message\s*:", query or ""))
                if kind == "hdfs_seq" and not is_hdfs_seq:
                    is_hdfs_seq = False
                if kind == "message" and not is_message:
                    is_message = False

                if is_hdfs_seq or is_message:
                    response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, seed_docs)
                    response = self.generate("", response_prompt)
                    if "gpt" in str(getattr(self.args, "generator", "")).lower() and response == "":
                        response = self.retry_loop("", response_prompt)
                    response = (response or "").strip()
                    reasoning = {
                        "question": query,
                        "mode": "tot",
                        "fastpath": False,
                        "skip_thought": True,
                        "aiops": True,
                        "aiops_query_kind": kind,
                        "nodes": [],
                    }
                    return response, seed_docs, [], reasoning
            except Exception:
                pass


        def dfs(question, history, supported, depth):
            layer_nodes = []
            logger.info("Search Depth:{}".format(depth))
            if depth > self.args.max_depth:
                return history, supported, layer_nodes
            if self.args.max_nodes is not None and depth <= len(self.args.max_nodes):
                response = self.retriever.get_documents(question=[question], top_k=self.args.max_nodes[depth-1])
            else:
                response = self.retriever.get_documents(question=[question], top_k=self.args.top_k_documents)
            for cnt, document in enumerate(response[0]["documents"]):
                logger.info("Thought of the {} node in depth {}.".format(cnt+1, depth))
                if not self.check_document_exist(supported=supported, document=document):
                    logger.warning("document exist in supported evidence path, skip and continue")
                    continue
                text = doc_to_text(document)
                history.append({
                    "id": document["id"],
                    "text": text,
                })
                history_docs = [his["text"] for his in history]
                thought_prompt = self.generator.get_thought_prompt(self.args.thought_shot, history_docs, query)
                response = self._generate_with_temp_max_len(self.generator, "", thought_prompt, thought_max_len)
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
                            tmp = {
                                "error_type": "Thought_prompt",
                                "prompt": thought_prompt,
                                "response": response
                            }
                            line = json.dumps(tmp, ensure_ascii=False)
                            f.write(line + "\n")
                            history.pop(-1)
                else:
                    self.success_thought += 1
                    if response_labels["decision"] == "reject":
                        history.pop(-1)
                    elif response_labels["decision"] == "continue":
                        new_question = response_labels["answer_content"]
                        history, supported, child_nodes = dfs(new_question, history, supported, depth+1)
                        leaf["child"] = child_nodes
                        history.pop(-1)
                    else:
                        supported.append({
                            "evidence": response_labels["answer_content"],
                            "history_docs": copy.deepcopy(history)})
                        history.pop(-1)

            return history, supported, layer_nodes

        history, supported, nodes = dfs(query, [], [], 1)

        evidence_tree = {
            "question": query,
            "nodes": nodes
        }

        docs_idx = []
        docs = []
        for supporting in supported:
            for document in supporting["history_docs"]:
                if document["id"] not in docs_idx:
                    docs_idx.append(document["id"])
                    docs.append(document["text"])

        response = _select_extractive_log_response(query, docs, response_mode=getattr(self.args, "log_response_mode", "line"))
        if response is None:
            response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, docs)
            response = self.generate("", response_prompt)
            if "gpt" in self.args.generator and response == "":
                response = self.retry_loop("", response_prompt)
        response = _maybe_truncate_to_target_prefix(self.args, query, docs, response)
        if response:
            response = response.splitlines()[0].strip()
        return response, docs, supported, evidence_tree

    def tree_of_thought_without_fusion_distribute(self, query):

        def dfs(question, history, supported, depth):
            layer_nodes = []
            if dist_utils.get_rank() == 0:
                logger.info("Search Depth:{}".format(depth))
            if depth > self.args.max_depth:
                return history, supported, layer_nodes
            if self.args.max_nodes is not None and depth <= len(self.args.max_nodes):
                response = self.retriever.get_documents(question=[question], top_k=self.args.max_nodes[depth-1])
            else:
                response = self.retriever.get_documents(question=[question], top_k=self.args.top_k_documents)
            for cnt, document in enumerate(response[0]["documents"]):
                if dist_utils.get_rank() == 0:
                    logger.info("Thought of the {} node in depth {}.".format(cnt+1, depth))
                if not self.check_document_exist(supported=supported, document=document):
                    if dist_utils.get_rank() == 0:
                        logger.warning("document exist in supported evidence path, skip and continue")
                    continue
                text = doc_to_text(document)
                history.append({
                    "id": document["id"],
                    "text": text,
                })
                history_docs = [his["text"] for his in history]
                thought_prompt = self.generator.get_thought_prompt(self.args.thought_shot, history_docs, query)
                dist_utils.barrier()
                if not dist.is_initialized() or dist_utils.get_rank() == 0:
                    response = self.generate("", thought_prompt)
                    if "gpt" in self.args.generator and response == "":
                        response = self.retry_loop("", thought_prompt)
                    if dist.is_initialized():
                        dist.broadcast_object_list([response], src=0)
                else:
                    response_list = [None]
                    dist.broadcast_object_list(response_list, src=0)
                    response = response_list[0]
                dist_utils.barrier()
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
                    self.consecutive_rejects += 1
                    if self.args.failed_parse_file is not None and dist_utils.get_rank() == 0:
                        with open(self.args.failed_parse_file, "a") as f:
                            tmp = {
                                "error_type": "Thought_prompt",
                                "prompt": thought_prompt,
                                "response": response
                            }
                            line = json.dumps(tmp, ensure_ascii=False)
                            f.write(line + "\n")
                    history.pop(-1)
                    # Smart pruning: early termination on consecutive failures
                    if self.consecutive_rejects >= self.max_consecutive_rejects:
                        if dist_utils.get_rank() == 0:
                            logger.warning(f"Consecutive rejects reached {self.consecutive_rejects}, early stopping.")
                        break
                else:
                    self.success_thought += 1
                    if response_labels["decision"] == "reject":
                        self.consecutive_rejects += 1
                        history.pop(-1)
                        # Smart pruning: too many consecutive rejections
                        if self.consecutive_rejects >= self.max_consecutive_rejects:
                            if dist_utils.get_rank() == 0:
                                logger.warning(f"Consecutive rejects reached {self.consecutive_rejects}, pruning branch.")
                            break
                    elif response_labels["decision"] == "continue":
                        # Reset rejection count (valid path found)
                        self.consecutive_rejects = 0
                        self.total_queries_generated += 1
                        # Improved query extraction logic
                        new_question = self.extract_query_content(response_labels["answer_content"])
                        if dist_utils.get_rank() == 0:
                            logger.info(f"Generated new query (#{self.total_queries_generated}): {new_question[:100]}...")
                        dist_utils.barrier()
                        history, supported, child_nodes = dfs(new_question, history, supported, depth+1)
                        leaf["child"] = child_nodes
                        history.pop(-1)
                    else:
                        # Answer found, reset counter
                        self.consecutive_rejects = 0
                        supported.append({
                            "evidence": response_labels["answer_content"],
                            "history_docs": copy.deepcopy(history)})
                        history.pop(-1)

            return history, supported, layer_nodes

        history, supported, nodes = dfs(query, [], [], 1)

        dist_utils.barrier()

        evidence_tree = {
            "question": query,
            "nodes": nodes
        }

        docs_idx = []
        docs = []
        for supporting in supported:
            for document in supporting["history_docs"]:
                if document["id"] not in docs_idx:
                    docs_idx.append(document["id"])
                    docs.append(document["text"])

        response_prompt = None
        response_candidate = _select_extractive_log_response(
            query, docs, response_mode=getattr(self.args, "log_response_mode", "line")
        )
        if not dist.is_initialized() or dist_utils.get_rank() == 0:
            if response_candidate is not None:
                response = response_candidate
            else:
                response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, docs)
                response = self.generate("", response_prompt)
                if "gpt" in self.args.generator and response == "":
                    response = self.retry_loop("", response_prompt)
            response = _maybe_truncate_to_target_prefix(self.args, query, docs, response)
            if response:
                response = response.splitlines()[0].strip()
            if dist.is_initialized():
                dist.broadcast_object_list([response], src=0)
        else:
            response_list = [None]
            dist.broadcast_object_list(response_list, src=0)
            response = response_list[0]

        dist_utils.barrier()
        return response, docs, supported, evidence_tree

    def tree_of_thought_missing_evidence_without_fusion_distribute(self, query):

        def dfs(question, history, supported, depth):
            layer_nodes = []
            if dist_utils.get_rank() == 0:
                logger.info("Search Depth:{}".format(depth))
            if depth > self.args.max_depth:
                return history, supported, layer_nodes
            if self.args.max_nodes is not None and depth <= len(self.args.max_nodes):
                response = self.retriever.get_documents(question=[question], top_k=self.args.max_nodes[depth-1])
            else:
                response = self.retriever.get_documents(question=[question], top_k=self.args.top_k_documents)
            for cnt, document in enumerate(response[0]["documents"]):
                if dist_utils.get_rank() == 0:
                    logger.info("Thought of the {} node in depth {}.".format(cnt+1, depth))
                if not self.check_document_exist(supported=supported, document=document):
                    if dist_utils.get_rank() == 0:
                        logger.warning("document exist in supported evidence path, skip and continue")
                    continue
                text = doc_to_text(document)
                history.append({
                    "id": document["id"],
                    "text": text,
                })
                history_docs = [his["text"] for his in history]
                thought_prompt = self.generator.get_thought_prompt(self.args.thought_shot, history_docs, query)
                dist_utils.barrier()
                if not dist.is_initialized() or dist_utils.get_rank() == 0:
                    response = self.generate("", thought_prompt)
                    if "gpt" in self.args.generator and response == "":
                        response = self.retry_loop("", thought_prompt)
                    if dist.is_initialized():
                        dist.broadcast_object_list([response], src=0)
                else:
                    response_list = [None]
                    dist.broadcast_object_list(response_list, src=0)
                    response = response_list[0]
                dist_utils.barrier()
                response_labels = self.parsing_thought(response)
                leaf = {
                    "depth": depth,
                    "nid": cnt,
                    "inputs": thought_prompt,
                    "outputs": response,
                    "labels": response_labels,
                    "state": "Failed" if response_labels is None else "Success",
                    "child": [],
                    "missing_evidence": None,
                    "missing_evidence_label": None,
                }
                layer_nodes.append(leaf)
                if response_labels is None:
                    self.failed_thought += 1
                    self.consecutive_rejects += 1
                    if self.args.failed_parse_file is not None and dist_utils.get_rank() == 0:
                        with open(self.args.failed_parse_file, "a") as f:
                            tmp = {
                                "error_type": "Thought_prompt",
                                "prompt": thought_prompt,
                                "response": response
                            }
                            line = json.dumps(tmp, ensure_ascii=False)
                            f.write(line + "\n")
                    history.pop(-1)
                    # Smart pruning: early termination on consecutive failures
                    if self.consecutive_rejects >= self.max_consecutive_rejects:
                        if dist_utils.get_rank() == 0:
                            logger.warning(f"🚫 Consecutive parse failures reached {self.consecutive_rejects}, early stopping.")
                        break
                else:
                    self.success_thought += 1
                    if response_labels["decision"] == "reject":
                        self.consecutive_rejects += 1
                        history.pop(-1)
                        # Smart pruning: too many consecutive rejections
                        if self.consecutive_rejects >= self.max_consecutive_rejects:
                            if dist_utils.get_rank() == 0:
                                logger.warning(f"🚫 Consecutive rejects reached {self.consecutive_rejects}, pruning branch.")
                            break
                    elif response_labels["decision"] == "continue":
                        # Reset rejection count (valid path found)
                        self.consecutive_rejects = 0
                        self.total_queries_generated += 1
                        history.pop(-1)
                    elif response_labels["decision"] == "continue":
                        missing_evidence_thought = self.generator.get_missing_evidence_prompt(self.args.missing_evidence_shot, query, history_docs)
                        dist_utils.barrier()
                        if not dist.is_initialized() or dist_utils.get_rank() == 0:
                            response = self.generate("", missing_evidence_thought)
                            if "gpt" in self.args.generator and response == "":
                                response = self.retry_loop("", missing_evidence_thought)
                            if dist.is_initialized():
                                dist.broadcast_object_list([response], src=0)
                        else:
                            response_list = [None]
                            dist.broadcast_object_list(response_list, src=0)
                            response = response_list[0]
                        dist_utils.barrier()
                        missing_evidence_label = self.parsing_missing_evidence(response)
                        leaf["missing_evidence_label"] = missing_evidence_label
                        if missing_evidence_label is None:
                            if self.args.failed_parse_file is not None and dist_utils.get_rank() == 0:
                                with open(self.args.failed_parse_file, "a") as f:
                                    tmp = {
                                        "error_type": "misssing_evidennce_prompt",
                                        "prompt": missing_evidence_thought,
                                        "response": response,
                                    }
                                    line = json.dumps(tmp, ensure_ascii=False)
                                    f.write(line + "\n")
                            # Use simplified query extraction
                            new_question = self.extract_query_content(response_labels["answer_content"])
                            leaf["missing_evidence"] = new_question
                            if dist_utils.get_rank() == 0:
                                logger.info(f"🔍 Generated query from thought (#{self.total_queries_generated}): {new_question[:100]}...")
                            history, supported, child_nodes = dfs(new_question, history, supported, depth+1)
                        else:
                            new_question = self.extract_query_content(missing_evidence_label["information"])
                            leaf["missing_evidence"] = new_question
                            if dist_utils.get_rank() == 0:
                                logger.info(f"🔍 Generated query from missing_evidence (#{self.total_queries_generated}): {new_question[:80]}...")
                            history, supported, child_nodes = dfs(new_question, history, supported, depth + 1)
                        leaf["child"] = child_nodes
                        history.pop(-1)
                    else:
                        # Answer found (anomaly judgment), reset counter
                        self.consecutive_rejects = 0
                        if dist_utils.get_rank() == 0:
                            logger.info(f"✅ Anomaly detection completed at depth {depth}")
                        supported.append({
                            "evidence": response_labels["answer_content"],
                            "history_docs": copy.deepcopy(history)})
                        history.pop(-1)

            return history, supported, layer_nodes

        history, supported, nodes = dfs(query, [], [], 1)

        dist_utils.barrier()

        evidence_tree = {
            "question": query,
            "nodes": nodes
        }

        docs_idx = []
        docs = []
        for supporting in supported:
            for document in supporting["history_docs"]:
                if document["id"] not in docs_idx:
                    docs_idx.append(document["id"])
                    docs.append(document["text"])

        response_prompt = None
        response_candidate = _select_extractive_log_response(
            query, docs, response_mode=getattr(self.args, "log_response_mode", "line")
        )
        if not dist.is_initialized() or dist_utils.get_rank() == 0:
            if response_candidate is not None:
                response = response_candidate
            else:
                response_prompt = self.generator.get_response_prompt(self.args.response_shot, query, docs)
                response = self.generate("", response_prompt)
                if "gpt" in self.args.generator and response == "":
                    response = self.retry_loop("", response_prompt)
            if dist.is_initialized():
                dist.broadcast_object_list([response], src=0)
        else:
            response_list = [None]
            dist.broadcast_object_list(response_list, src=0)
            response = response_list[0]

        dist_utils.barrier()
        return response, docs, supported, evidence_tree
