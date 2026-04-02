import json
import logging
import re

import dist_utils
import torch.distributed as dist

logger = logging.getLogger(__name__)


class FusionMixin:
    def post_evidence_fusion(self, data):
        evidence = data["evidence"]
        evidence = [e["evidence"] for e in evidence]
        question = data["question"]
        fusion_prompt = self.generator.get_evidence_fusion_prompt(shot=self.args.evidence_fusion_shot, evidence=evidence, question=question)
        response = self.generate("", fusion_prompt)
        if "gpt" in self.args.generator and response == "":
            response = self.retry_loop("", fusion_prompt)
        response_label = self.parsing_fusion(response)
        categories = []
        if response_label is None:
            if self.args.failed_parse_file is not None:
                with open(self.args.failed_parse_file, "a") as f:
                    tmp = {
                        "error_type": "Fusion_prompt",
                        "prompt": fusion_prompt,
                        "response": response,
                        "data": data,
                    }
                    line = json.dumps(tmp, ensure_ascii=False)
                    f.write(line + "\n")
                    logging.warning("WARNING: Invalid parse in fusion, please pay attention!")
                    return None
        for label in response_label:
            docs = []
            docs_id = []
            for idx in label["index"]:
                if idx-1 < len(evidence):
                    document = data["evidence"][idx-1]["history_docs"]
                    for d in document:
                        if d["id"] not in docs_id:
                            docs.append(d)
                            docs_id.append(d["id"])
            categories.append({
                "opinion": label["opinion"],
                "documents": docs
            })
        return categories

    def evidence_fusion(self, supported, history, query):
        def get_conflict_evidence(judgement, supporting):
            import re
            result = re.findall(r'\d+', judgement)
            if len(result) != 1:
                return -1
            if int(result[0]) > len(supporting):
                return -1
            return int(result[0]) - 1

        system_prompt, summary_prompt = self.generator.get_evidence_summary_prompt(self.args.evidence_summary_shot, query, history)
        response = self.generator(system_prompt, summary_prompt)
        candidate = response.strip()
        if len(supported) == 0:
            return [{
                "evidence": candidate,
                "history_docs": copy.deepcopy(history)
            }]
        system_prompt, fusion_prompt = self.generator.get_evidence_fusion_prompt(self.args.evidence_fusion_shot, supported, candidate)
        response = self.generator(system_prompt, fusion_prompt)
        choice = response.strip().split("\n")[-1]
        choice = choice.lower()
        if "accept" in choice:
            supported.append({
                "evidence": candidate,
                "history_docs": copy.deepcopy(history)
            })
            return supported
        elif "repetition" in choice:
            return supported
        else:
            conflict_evidence_idx = get_conflict_evidence(choice, supported)
            if conflict_evidence_idx == -1:
                supported.append({
                    "evidence": candidate,
                    "history_docs": copy.deepcopy(history)
                })
                return supported
            origin_evidence = supported[conflict_evidence_idx]
            conflict_evidence = [supported[conflict_evidence_idx]["evidence"], candidate]
            supported.pop(conflict_evidence_idx)
            system_prompt, query_prompt = self.generator.get_conflict_evidence_prompt(self.args.conflict_evidence_shot, conflict_evidence)
            conflict_solution = self.generator(system_prompt, query_prompt)
            conflict_solution_relevant_docs = self.retriever.get_documents(question=[conflict_solution], top_k=self.args.top_k_documents)
            documents = []
            for document in conflict_solution_relevant_docs["documents"]:
                text = doc_to_text(document)
                documents.append(text)
            system_prompt, fusion_prompt = self.generator.get_conflict_fusion_prompt(self.conflict_fusion_shot, conflict_evidence, documents)
            judgement = self.generator(system_prompt, fusion_prompt)
            judgement = judgement.lower()
            if "accept first" in judgement:
                supported.append(origin_evidence)
                return supported
            elif "accept second" in judgement:
                supported.append({
                    "evidence": conflict_evidence[1],
                    "history_docs": copy.deepcopy(history),
                })
                return supported
            else:
                supported.append(origin_evidence)
                supported.append({
                    "evidence": conflict_evidence[1],
                    "history_docs": copy.deepcopy(history),
                })
                return supported
