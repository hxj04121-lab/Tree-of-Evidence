import re
import logging

logger = logging.getLogger(__name__)


class ParsingMixin:
    def parsing_thought(self, response, gate_mode: str = None):
        raw_response = response or ""
        input_str = raw_response.lower()
        mode = str(gate_mode or getattr(self.args, "gate_mode", "full") or "full").strip().lower()
        if mode not in {"full", "vr_only", "no_gate"}:
            mode = "full"
        try:
            # Step 4 may be truncated. If steps 1-3 are present, attempt partial parse.
            has_s4 = 'step 4' in input_str
            if 'step 1' not in input_str:
                self.logger_rank0(info="Error: Missing step 1")
                return None
            if not has_s4 and 'step 2' not in input_str:
                self.logger_rank0(info="Error: Missing step 2 and step 4")
                return None

            if 'step 2' in input_str and 'step 3' in input_str:
                step1_index = input_str.index('step 1')
                step2_index = input_str.index('step 2')
                step3_index = input_str.index('step 3')
                step4_index = input_str.index('step 4') if has_s4 else len(input_str)
                if step1_index > step2_index or step2_index > step3_index or step3_index > step4_index:
                    self.logger_rank0(info="Error: Subscript in a wrong order!!")
                    return None

                step1_str = input_str[step1_index:step2_index]
                step2_str = input_str[step2_index:step3_index]
                step3_str = input_str[step3_index:step4_index]
                step4_str = input_str[step4_index:]

                if '[relevant]' not in step1_str and '[irrelevant]' not in step1_str:
                    self.logger_rank0(info="Error: Missing [relevant] or [irrelevant] in step1")
                    return None
                if '[relevant]' in step1_str:
                    relevant = 'relevant'
                else:
                    relevant = 'irrelevant'

                if '[supported]' not in step2_str and '[unsupported]' not in step2_str:
                    self.logger_rank0(info="Error: Missing [supported] or [unsupported] in step2")
                    return None
                if '[supported]' in step2_str:
                    support = 'supported'
                else:
                    support = 'unsupported'

                if '[answer]' not in step3_str and '[query]' not in step3_str:
                    self.logger_rank0(info="Error: Missing [answer] or [query] in step3")
                    return None
                if '[answer]' in step3_str:
                    answer = 'answer'
                    answer_content = step3_str[step3_str.rfind('[answer]') + 8:]
                else:
                    answer = 'query'
                    answer_content = step3_str[step3_str.rfind('[query]') + 7:]

                step4_decision = None
                if '[accepted]' in step4_str:
                    step4_decision = 'accepted'
                elif '[continue]' in step4_str:
                    step4_decision = 'continue'
                elif '[reject]' in step4_str:
                    step4_decision = 'reject'

                # Gate ablations (Task 1.4)
                if mode == "full":
                    if relevant == "irrelevant":
                        decision = "reject"
                    elif answer == "answer" and support == "supported":
                        decision = "accepted"
                    elif answer == "query" and support == "unsupported":
                        decision = "continue"
                    else:
                        decision = "reject"
                elif mode == "vr_only":
                    if relevant == "irrelevant":
                        decision = "reject"
                    elif answer == "answer":
                        decision = "accepted"
                    elif answer == "query":
                        decision = "continue"
                    else:
                        decision = step4_decision or "reject"
                else:
                    # no_gate: trust model's Step 4 when present; otherwise fall back to Step 3.
                    if step4_decision is not None:
                        decision = step4_decision
                    elif answer == "answer":
                        decision = "accepted"
                    elif answer == "query":
                        decision = "continue"
                    else:
                        decision = "reject"

                return {
                    'relevant': relevant,
                    'support': support,
                    'answer': answer,
                    'answer_content': answer_content,
                    'decision': decision,
                    'gate_mode': mode,
                    'step4_decision': step4_decision,
                }

            if "step 2" not in input_str and "step 3" not in input_str:
                step1_index = input_str.index('step 1')
                step4_index = input_str.index('step 4')
                if step1_index > step4_index:
                    self.logger_rank0(info="Error: Subscript in a wrong order!!")
                    return None
                step1_str = input_str[step1_index:step4_index]
                if '[relevant]' not in step1_str and '[irrelevant]' not in step1_str:
                    self.logger_rank0(info="Error: Missing [relevant] or [irrelevant] in step1")
                    return None
                if '[relevant]' in step1_str:
                    self.logger_rank0(info="Error: Missing required substrings")
                    return None

                return {'relevant': "irrelevant", 'support': None, 'answer': None, 'answer_content': None,
                        'decision': "reject"}
        except:
            pass

        # Fallback: scan full text for gate keywords (handles DeepSeek free-form output).
        try:
            relevant = ('relevant' if '[relevant]' in input_str
                        else ('irrelevant' if '[irrelevant]' in input_str else None))
            support = ('supported' if '[supported]' in input_str
                       else ('unsupported' if '[unsupported]' in input_str else None))
            answer = ('answer' if '[answer]' in input_str
                      else ('query' if '[query]' in input_str else None))
            step4_decision = ('accepted' if '[accepted]' in input_str
                              else ('continue' if '[continue]' in input_str
                                    else ('reject' if '[reject]' in input_str else None)))
            if relevant is None and step4_decision is None:
                self.logger_rank0(info="Error: Fallback scan found no gate keywords")
                return None
            # Derive decision from keywords.
            if relevant == 'irrelevant' or step4_decision == 'reject':
                decision = 'reject'
            elif step4_decision == 'accepted' or (answer == 'answer' and support == 'supported'):
                decision = 'accepted'
            elif step4_decision == 'continue' or answer == 'query':
                decision = 'continue'
            else:
                decision = 'reject'
            answer_content = ''
            if answer == 'answer' and '[answer]' in input_str:
                answer_content = input_str[input_str.rfind('[answer]') + 8:].strip()
            elif answer == 'query' and '[query]' in input_str:
                answer_content = input_str[input_str.rfind('[query]') + 7:].strip()
            self.logger_rank0(info=f"Fallback parse: relevant={relevant} support={support} answer={answer} decision={decision}")
            return {
                'relevant': relevant or 'relevant',
                'support': support,
                'answer': answer,
                'answer_content': answer_content,
                'decision': decision,
                'gate_mode': mode,
                'step4_decision': step4_decision,
            }
        except:
            pass
        # Fallback 2: infer a minimal gate decision from DeepSeek free-form prose.
        try:
            def _has_any(text, phrases):
                return any(p in text for p in phrases)

            relevant = None
            if _has_any(input_str, [
                "irrelevant", "not relevant", "unrelated", "different component",
                "different module", "no logical connection",
            ]):
                relevant = "irrelevant"
            elif _has_any(input_str, [
                "relevant", "related", "same block", "same component",
                "same module", "same host", "same node",
            ]):
                relevant = "relevant"

            support = None
            if _has_any(input_str, [
                "insufficient evidence", "not enough evidence", "evidence is insufficient",
                "need more evidence", "need more information", "need additional information",
                "ambiguous", "uncertain", "cannot determine",
            ]):
                support = "unsupported"
            elif _has_any(input_str, [
                "sufficient evidence", "evidence is sufficient", "enough evidence",
                "clear evidence", "evidence supports",
            ]):
                support = "supported"

            answer = None
            answer_content = ""
            step4_decision = None

            if re.search(r"(?i)\bjudgment\s*:\s*anomaly\b", raw_response) or _has_any(input_str, [
                "query indicates an anomaly", "this is an anomaly", "the sequence is an anomaly",
                "classify this as anomaly", "classification is anomaly",
            ]):
                answer = "answer"
                answer_content = "Anomaly"
                support = support or "supported"
                step4_decision = "accepted"
            elif re.search(r"(?i)\bjudgment\s*:\s*normal\b", raw_response) or _has_any(input_str, [
                "query indicates normal", "this is normal", "the sequence is normal",
                "classify this as normal", "classification is normal",
            ]):
                answer = "answer"
                answer_content = "Normal"
                support = support or "supported"
                step4_decision = "accepted"
            elif _has_any(input_str, [
                "search for", "find more logs", "need more logs", "query for",
                "retrieve more", "look for additional",
            ]):
                answer = "query"
                answer_content = raw_response.strip()
                support = support or "unsupported"
                step4_decision = "continue"

            if relevant is None and (support is not None or answer is not None or step4_decision is not None):
                relevant = "relevant"

            if relevant is None and step4_decision is None:
                raise ValueError("No heuristic gate signals found")

            if step4_decision == "reject" or relevant == "irrelevant":
                decision = "reject"
            elif step4_decision == "continue" or answer == "query":
                decision = "continue"
            elif step4_decision == "accepted" or answer == "answer":
                decision = "accepted"
            else:
                decision = "reject"

            self.logger_rank0(
                info=f"Heuristic parse: relevant={relevant} support={support} answer={answer} decision={decision}"
            )
            return {
                'relevant': relevant,
                'support': support,
                'answer': answer,
                'answer_content': answer_content,
                'decision': decision,
                'gate_mode': mode,
                'step4_decision': step4_decision,
            }
        except:
            pass
        self.logger_rank0(info="Error: Missing required substrings")
        return None

    def parsing_missing_evidence(self, response):
        input_str = response.lower()
        try:
            if 'step 1' not in input_str or 'step 2' not in input_str:
                self.logger_rank0(info="Error: Missing required substrings")
                return None
            step1_index = input_str.index('step 1')
            step2_index = input_str.index('step 2')
            if step1_index > step2_index:
                self.logger_rank0(info="Error: Subscript in a wrong order!!")
                return None
            step1_str = input_str[step1_index:step2_index]
            step2_str = input_str[step2_index:]
            if '[info]' not in step1_str:
                self.logger_rank0(info="Error: Missing [info] in step1")
                return None
            else:
                information = step1_str[step1_str.rfind('[info]') + 6:]
            if '[answer]' not in step2_str:
                self.logger_rank0(info="Error: Missing [answer] in step2")
                return None
            else:
                answer = step2_str[step2_str.rfind('[answer]') + 8:]
            return {'information': information, 'answer': answer}
        except:
            return None

    def parsing_fusion(self, response):
        parts = []
        cnt = 1
        pattern = "Opinion {}:".format(cnt)
        while pattern in response:
            cnt += 1
            new_pattern = "Opinion {}:".format(cnt)
            index = response.index(pattern)
            if new_pattern not in response:
                parts.append(response[index:])
            else:
                new_index = response.index(new_pattern)
                if index > new_index:
                    logger.warning("Error: Opinion not in a correct order!!")
                    return None
                parts.append(response[index:new_index])
            pattern = new_pattern
        result = []
        for cnt, part in enumerate(parts):
            opinion_idx = "Opinion {}:".format(cnt+1)
            index_idx = "Index {}:".format(cnt+1)
            if opinion_idx not in part or index_idx not in part:
                logger.warning("Error: Missing required substrings")
                return None
            opinion_idx = part.index(opinion_idx)
            index_idx = part.index(index_idx)
            if opinion_idx > index_idx:
                logger.warning("Error: Opinion not in a correct order!!")
            opinion = part[opinion_idx + 10:index_idx].strip()
            index = part[index_idx + 8:].strip()
            index = re.findall(r'\d+', index)
            try:
                index = [int(idx) for idx in index]
            except:
                logger.warning("Error: Index extraction error!!")
                return None
            result.append({
                "opinion": opinion,
                "index": index,
            })
        return result
