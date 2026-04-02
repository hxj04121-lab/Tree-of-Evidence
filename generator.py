import argparse
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Conditional import for llama to avoid errors when not using Llama models
try:
    from llama import Llama, Dialog
except ImportError:
    Llama = None
    Dialog = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_max_memory():
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


class GenerateLanguageModel:
    def __init__(self, args):
        self.args = args
        self.called_times = 0
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.repeat_times = 3
        # --- Cost/latency instrumentation (Task 1.2) ---
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_log: list = []  # per-call records: {prompt_tokens, completion_tokens, latency_ms}

    def reset_cost_counters(self):
        """Reset counters for per-sample measurement."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_log = []
        self.called_times = 0

    def get_cost_snapshot(self) -> dict:
        """Return current cost counters."""
        return {
            "llm_calls": self.called_times,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "call_log": list(self.call_log),
        }

    def get_model(self):
        pass

    def get_inputs(self, queries, system_prompt):
        pass

    def get_response(self, queries, system_prompt):
        pass

    def get_demo(self, demo_data, shot):
        pos_shot = (shot + 1) // 2
        neg_shot = shot - pos_shot
        
        # Handle cases where demo data is empty or insufficient
        demo_relevant = demo_data.get("demo_relevant", [])
        demo_irrelevant = demo_data.get("demo_irrelevant", [])
        
        # Adjust shot sizes if demos are insufficient
        pos_shot = min(pos_shot, len(demo_relevant))
        neg_shot = min(neg_shot, len(demo_irrelevant))
        
        pos_demo = random.sample(demo_relevant, pos_shot) if demo_relevant else []
        neg_demo = random.sample(demo_irrelevant, neg_shot) if demo_irrelevant else []
        
        pos_sample = []
        for demo in pos_demo:
            doc_idx = random.randint(0, len(demo["documents"]) - 1)
            pos_sample.append("\nQuestion:" + demo["question"] + "\nDocument:" + demo["documents"][doc_idx]["document"]["text"]
                      + "\nAnswer:" + demo["documents"][doc_idx]["answer"])
            
        neg_sample = []
        for demo in neg_demo:
            doc_idx = random.randint(0, len(demo["documents"]) - 1)
            neg_sample.append("\nQuestion:" + demo["question"] + "\nDocument:" + demo["documents"][doc_idx]["document"]["text"]
                      + "\nAnswer:" + demo["documents"][doc_idx]["answer"])
                      
        pos_sample.extend(neg_sample)
        return pos_sample

    """
    prompt format
        "Instruct:{INST}\n\nDemonstration:{DEMO}\n\nQuestion:{Q}\n\nDocuments: {D}"
    """
    def get_thought_prompt(self, shot=0, documents=None, question=None):
        with open(self.args.thought_config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]
        documents = "\n".join(documents)
        if shot > 0:
            demos = self.get_demo(config, shot=shot)
            if demos:  # Only add demos if they exist
                demos = "\n".join(demos)
                prompt = prompt.replace("{INST}", instrution)
                prompt = prompt.replace("{DEMO}", demos)
                prompt = prompt.replace("{D}", documents)
                prompt = prompt.replace("{Q}", question)
            else:
                # No demos available, remove {DEMO} placeholder
                prompt = prompt.replace("{INST}", instrution)
                prompt = prompt.replace("{DEMO}\n\n", "")
                prompt = prompt.replace("{D}", documents)
                prompt = prompt.replace("{Q}", question)
        else:
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}\n\n", "")
            prompt = prompt.replace("{D}", documents)
            prompt = prompt.replace("{Q}", question)
        return prompt

    """
    prompt format
        ""Instruct:{INST}\n\nDemonstration:{D}\n\nEvidences: {E}\n\nQuestion:{Q}""
    """
    def get_evidence_fusion_prompt(self, shot=0, evidence=None, question=None):
        with open(self.args.evidence_fusion_config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]
        evidence = "\n".join(evidence)
        if shot > 0:
            demo_data = config["demo"]
            demo_data = random.sample(demo_data, shot)
            demo_data = ["Evidence:\n" + d["evidence"] + "Question:\n" + d["question"] + "\n" +
                         d["answer"]for d in demo_data]
            demos = "\n".join(demo_data)
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{D}", demos)
            prompt = prompt.replace("{E}", evidence)
            prompt = prompt.replace("{Q}", question)
        else:
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{D}\n\n", "")
            prompt = prompt.replace("{E}", evidence)
            prompt = prompt.replace("{Q}", question)
        return prompt

    """
    prompt format
        "{INST}\n\n{DEMO}\n\nQuestion: {Q}\n\n Documents: {D}\n\nAnswer:{A}"
    """
    def get_evidence_summary_prompt(self, shot=0, question=None, documents=None):
        pass

    """
    prompt format
        "Instruct:{INST}\n\nDocuments: {D}\n\nQuestion:{Q}"
    """
    def get_response_prompt(self, shot=0, question=None, documents=None):
        with open(self.args.response_config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]
        documents = ["[{}]".format(cnt+1) + doc for cnt, doc in enumerate(documents)]
        max_doc_chars = int(getattr(self.args, "prompt_doc_max_chars", 0) or 0)
        if max_doc_chars > 0:
            clipped = []
            for d in documents:
                s = str(d)
                if len(s) > max_doc_chars:
                    s = s[:max_doc_chars].rstrip() + " ..."
                clipped.append(s)
            documents = clipped

        documents = "\n".join(documents)
        
        prompt = prompt.replace("{INST}", instrution)
        if "{DEMO}" in prompt:
             # Future work: Implement get_demo for response prompt if needed
             prompt = prompt.replace("{DEMO}", "")
             # Clean up extra newlines if {DEMO} was removed
             prompt = prompt.replace("\n\n\n\n", "\n\n")

        prompt = prompt.replace("{D}", documents)
        q = question or ""
        max_q_chars = int(getattr(self.args, "prompt_query_max_chars", 0) or 0)
        if max_q_chars > 0 and len(q) > max_q_chars:
            q = q[:max_q_chars].rstrip() + " ..."
        prompt = prompt.replace("{Q}", q)
        return prompt

    """
    prompt format
        "Instruct:{INST}\n\nQuestion:{Q}\n\nCandidates:\n{C}\n\nAnswer:"
    """
    def get_vote_prompt(self, question=None, candidates=None):
        path = getattr(self.args, "block_chain_vote_judge_prompt_path", None) or "prompts/vote_prompt.json"
        with open(path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instruction = config.get("instruct", "")
        prompt = config.get("format", "Instruct:{INST}\n\nQuestion:{Q}\n\nCandidates:\n{C}\n\nAnswer:")

        lines = []
        for i, c in enumerate(candidates or [], start=1):
            block_id = str(c.get("block_id") or "").strip()
            answer = str(c.get("answer") or "").strip().replace("\r", " ").replace("\n", " ")
            score = c.get("score", None)
            score_str = ""
            if score is not None:
                try:
                    score_str = f"{float(score):.4f}"
                except Exception:
                    score_str = str(score)

            meta = []
            if block_id:
                meta.append(f"block={block_id}")
            if score_str:
                meta.append(f"score={score_str}")
            meta = " ".join(meta).strip()

            if meta:
                lines.append(f"[{i}] {meta} answer: {answer}")
            else:
                lines.append(f"[{i}] {answer}")

        cand_text = "\n".join(lines) if lines else "(no candidates)"
        prompt = prompt.replace("{INST}", instruction)
        prompt = prompt.replace("{Q}", question or "")
        prompt = prompt.replace("{C}", cand_text)
        return prompt

    def get_hotpotqa_response_prompt(self, shot=0, question=None, documents=None):
        with open(self.args.response_config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]

        documents = ["[{}]".format(cnt + 1) + doc for cnt, doc in enumerate(documents)]
        documents = "\n".join(documents)
        if shot > 0:
            demos = random.sample(config["demos"], shot)
            sample = []
            for demo in demos:
                demo_docs = ["[{}]".format(cnt + 1) + doc for cnt, doc in enumerate(demo["documents"])]
                demo_docs = "\n".join(demo_docs)
                sample.append("\nQuestion:" + demo["question"] + "\nDocument:" + demo_docs + "\n" + demo["answer"])

            demonstration = config["delimiter"].join(sample)

            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}", demonstration)
            prompt = prompt.replace("{D}", documents)
            prompt = prompt.replace("{Q}", question)
        else:
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}\n\n", "")
            prompt = prompt.replace("{D}", documents)
            prompt = prompt.replace("{Q}", question)

        return prompt

    def get_hotpotqa_baseline_response_prompt(self, shot=0, question=None, documents=None):
        with open(self.args.response_config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]

        documents = ["[{}]".format(cnt + 1) + doc for cnt, doc in enumerate(documents)]
        documents = "\n".join(documents)
        if shot > 0:
            demos = random.sample(config["demos"], shot)

            demonstration = config["delimiter"].join(demos)

            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}", demonstration)
            prompt = prompt.replace("{D}", documents)
            prompt = prompt.replace("{Q}", question)
        else:
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}\n\n", "")
            prompt = prompt.replace("{D}", documents)
            prompt = prompt.replace("{Q}", question)

        return prompt
    """
    prompt format
        "{INST}\n\n{DEMO}\n\nConflict Evidences: {CE}\n\n Query: {Q}"
    """
    def get_conflict_evidence_prompt(self, shot=0, conflict=None):
        pass

    """
    prompt format
        "{INST}\n\n{DEMO}\n\nQuestion: {Q}\n\n References: {Ref}"
    """
    def get_missing_evidence_prompt(self, shot=0, question=None, history=None):
        def get_demo(config, shot):
            demo = config["demos"][:10]
            demo_sample = random.sample(demo, shot)
            demo_sample = ["Question:{}\nReferences:{}\n{}".format(d["question"], d["references"], d["response"]) for d in demo_sample]
            return demo_sample
        with open(self.args.missing_evidence_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        instrution = config["instruct"]
        prompt = config["format"]
        documents = "\n".join(history)
        if shot > 0:
            demos = get_demo(config, shot=shot)
            demos = "\n".join(demos)
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}", demos)
            prompt = prompt.replace("{Q}", question)
            prompt = prompt.replace("{Ref}", documents)
        else:
            prompt = prompt.replace("{INST}", instrution)
            prompt = prompt.replace("{DEMO}\n\n", "")
            prompt = prompt.replace("{Q}", question)
            prompt = prompt.replace("{Ref}", documents)
        return prompt

    """
    prompt format
        "{INST}\n\n{DEMO}\n\nConflict Evidences: {CE}\n\n Documents: {D} \n\n
         Judgement: {Accept First, Accept Second, Accept Both}"
    """
    def get_conflict_fusion_prompt(self, shot=0, conflict=None, documents=None):
        pass


class ChatGPTModel(GenerateLanguageModel):
    def __init__(self, args):
        super(ChatGPTModel, self).__init__(args)
        self.gpt35_sleep_time = 5
        self.gpt4_sleep_time = 20
        self.repeat_times = 6

        if self.args.generator == "gpt35":
            self.url = ""  # set your API endpoint
            self.headers = {
                "X-Gateway-Stage": "RELEASE",
                "X-Gateway-SecretId": "",
                "Content-Type": "application/json",
                "X-Gateway-SecretKey": "",
            }
            self.userID = ""

        elif self.args.generator == "gpt35_instruct":
            self.url = ""  # set your API endpoint
            self.headers = {
                "X-Gateway-Stage": "RELEASE",
                "X-Gateway-SecretId": "",
                "Content-Type": "application/json",
                "X-Gateway-SecretKey": "",
            }
            self.userID = ""
        elif self.args.generator == "gpt4":
            self.url = ""  # set your API endpoint
            self.headers = {
                "X-Gateway-Stage": "RELEASE",
                "X-Gateway-SecretId": "",
                "Content-Type": "application/json",
                "X-Gateway-SecretKey": "",
            }
            self.userID = ""
        else:
            print("You must name as gpt35 or gpt4 to use chatgpt api!!")

    def get_body(self, system_prompt, input):
        if system_prompt == "":
            system_prompt = "You are ChatGPT."
        if self.args.generator == "gpt35":
            body = {
                "userId": self.userID,
                "model": "gpt-3.5-turbo",
                "messages": [{
                    'role': 'system', 'content': system_prompt,
                }, {
                    'role': 'user', 'content': input,
                }
                ]
            }
        elif self.args.generator == "gpt35_instruct":
            body = {
                # "userId": self.userID,
                # "model": "gpt-3.5-turbo-instruct",
                "prompt": [input],
                "max_tokens": self.args.max_seq_len,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
            }

        elif self.args.generator == "gpt4":
            body = {
                "userId": self.userID,
                "model": "gpt-4",
                "set": "gpt4",
                "messages": [{
                    'role': 'system', 'content': system_prompt,
                }, {
                    'role': 'user', 'content': input,
                }
                ]
            }
        else:
            print("You must name as gpt35 or gpt4 to use chatgpt api!!")
        return body

    def get_response(self, body):
        if "gpt35" in self.args.generator:
            time.sleep(self.gpt35_sleep_time)
        else:
            time.sleep(self.gpt4_sleep_time)
        data = json.dumps(body).encode("utf-8")
        print("*" * 50)
        try:
            request = urllib.request.Request(url=self.url, data=data, headers=self.headers)
            response = urllib.request.urlopen(request, timeout=20 * 3600)
        except urllib.error.HTTPError as e:
            print("HTTP Error:", e.code, self.url)
            return ""
        except urllib.error.URLError as e:
            print("URL Error:", e.reason, self.url)
            return ""

        json_response = response.read().decode("utf-8")
        try:
            passages = json.loads(json_response)
            if self.args.generator == "gpt35_instruct":
                response = passages["data"]["choices"][0]["text"]
            else:
                response = passages["resp"]["choices"][0]["message"]["content"]
        except:
            print("Request failed")
            print(json_response)
            return ""
        response = response.strip()
        print(response)
        return response

class OpenAIChatGPTModel(GenerateLanguageModel):
    def __init__(self, args):
        super(OpenAIChatGPTModel, self).__init__(args)
        self.gpt35_sleep_time = 5
        self.gpt4_sleep_time = 30
        self.gpt35_list = ["text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k",
                           "gpt-3.5-turbo-0613", "gpt-3.5-turbo-instruct"]
        self.gpt4_list = ["gpt-4", "gpt-4-0314", "gpt-4-0613"]

        # Read API base URL from environment variable, default to proxy domain.
        # Compatible with two common formats:
        # - OPENAI_API_BASE=http://host:port
        # - OPENAI_API_BASE=http://host:port/v1
        api_base_url = (os.getenv("OPENAI_API_BASE", "https://api.openai.com") or "").rstrip("/")
        if api_base_url.endswith("/v1"):
            api_v1 = api_base_url
        else:
            api_v1 = f"{api_base_url}/v1"
        self.url = f"{api_v1}/chat/completions"

        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    def get_body(self, system_prompt, input):
        if not system_prompt:
            system_prompt = "You are a helpful assistant."

        # The codebase sometimes calls `get_response(queries=[prompt])` (a list),
        # so make sure we always send a plain string to OpenAI-compatible servers.
        if isinstance(system_prompt, (list, tuple)):
            system_prompt = "\n".join(str(x) for x in system_prompt if x is not None)
        if isinstance(input, (list, tuple)):
            if len(input) == 1:
                input = input[0]
            else:
                input = "\n".join(str(x) for x in input if x is not None)
        if input is None:
            input = ""
        if not isinstance(input, str):
            input = str(input)
        
        body = {
            "model": self.args.generator,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input}
            ],
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            # Limit generation length for local OpenAI-compatible servers (prevents very slow runs
            # when the model doesn't emit EOS promptly).
            "max_tokens": int(getattr(self.args, "max_gen_len", 256) or 256),
        }
        return body

    def get_response(self, body=None, queries=None, system_prompt=""):
        """
        OpenAI-compatible chat completion.

        Supports both call styles used in the codebase:
        - `get_response(body=...)` (legacy)
        - `get_response(queries=..., system_prompt=...)` (generator base interface)
        """
        if body is None:
            if queries is None:
                return ""
            body = self.get_body(system_prompt=system_prompt, input=queries)

        # Count each attempted API call (used by Task 1.2 cost instrumentation).
        self.called_times += 1

        print("*" * 50)
        _t0 = time.time()
        try:
            resp = requests.post(self.url, headers=self.headers, json=body, timeout=20 * 3600)
        except Exception as e:
            logger.warning("OpenAI API request failed: %s", e)
            return ""
        _latency_ms = (time.time() - _t0) * 1000

        if getattr(resp, "status_code", 0) != 200:
            preview = (getattr(resp, "text", "") or "")[:500]
            logger.warning("OpenAI API HTTP %s: %s", getattr(resp, "status_code", "?"), preview)
            return ""

        try:
            payload = resp.json()
        except Exception:
            logger.warning("OpenAI API returned non-JSON response: %s", (resp.text or "")[:500])
            return ""

        # --- Cost instrumentation: capture token usage ---
        _usage = payload.get("usage") or {}
        _pt = int(_usage.get("prompt_tokens", 0))
        _ct = int(_usage.get("completion_tokens", 0))
        self.total_prompt_tokens += _pt
        self.total_completion_tokens += _ct
        self.call_log.append({
            "prompt_tokens": _pt,
            "completion_tokens": _ct,
            "latency_ms": round(_latency_ms, 1),
        })

        text = ""
        try:
            choice0 = (payload.get("choices") or [None])[0] or {}
            if isinstance(choice0.get("message"), dict):
                text = choice0["message"].get("content", "") or ""
            else:
                text = choice0.get("text", "") or ""
        except Exception:
            text = ""

        text = (text or "").strip()
        if not text and payload.get("error"):
            logger.warning("OpenAI API error: %s", payload.get("error"))
        return text


class LLamaModel(GenerateLanguageModel):
    def __init__(self, args):
        super(LLamaModel, self).__init__(args)
        if Llama is None:
            raise RuntimeError("Llama module is not installed. Please install it to use LLamaModel.")
        self.model = self.get_model()

    def get_model(self):
        generator = Llama.build(
            ckpt_dir=self.args.generator_file_path,
            tokenizer_path=self.args.generator_tokenizer_path,
            max_seq_len=self.args.max_seq_len,
            max_batch_size=self.args.max_batch_size,
        )
        return generator

    def get_inputs(self, queries, system_prompt):
        inputs = []
        for query in queries:
            input_ = []
            if system_prompt != "":
                input_.append({"role": "system", "content": system_prompt})
            input_.append({"role": "user", "content": query})
            inputs.append(input_)
        return inputs

    def get_response(self, queries, system_prompt):
        inputs = []
        for query in queries:
            input_ = []
            if system_prompt != "":
                input_.append({"role": "system", "content": system_prompt})
            input_.append({"role": "user", "content": query})
            inputs.append(input_)

        results = self.model.chat_completion(
            inputs,
            max_gen_len=self.args.max_gen_len,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        for query, result in zip(queries, results):
            # print("query: " + query)
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )


class LLamaModelHF(GenerateLanguageModel):
    def __init__(self, args):
        super(LLamaModelHF, self).__init__(args)
        self.model, self.tokenizer = self.get_model()

    def get_model(self, dtype=torch.float16, int8=False):
        logger.info(f"Loading {self.args.generator_file_path} in {dtype}...")
        if int8:
            logger.warn("Use LLM.int8")
        start_time = time.time()
        generator = AutoModelForCausalLM.from_pretrained(
            self.args.generator_file_path,
            device_map='sequential',
            torch_dtype=dtype,
            max_memory=get_max_memory(),
            load_in_8bit=int8,
        )
        logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.generator_file_path, use_fast=False)

        # Fix OPT bos token problem in HF
        if "opt" in self.args.generator_file_path:
            tokenizer.bos_token = "<s>"
        tokenizer.padding_side = "left"

        return generator, tokenizer

    def get_inputs(self, queries, system_prompt):
        inputs = [system_prompt + "\n" + query for query in queries]
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        return inputs

    def get_response(self, queries, system_prompt):
        self.called_times += 1
        if system_prompt == "":
            inputs = queries
        else:
            inputs = [system_prompt + "\n" + query for query in queries]
        # inputs = queries

        prompt_len = len(self.tokenizer.tokenize(queries[0]))
        max_gen_len = min(self.args.max_seq_len - prompt_len, self.args.max_gen_len)
        if max_gen_len < 0:
            logger.warning("Prompt exceeds max length and return an empty string as answer. "
                           "If this happens too many times, it is suggested to make the prompt shorter")
            self.prompt_exceed_max_length += 1
            with open(self.args.failed_parse_file, "a") as f:
                tmp = {
                    "error_type": "Prompt_exceed_max_length",
                    "prompt": queries[0],
                    "response": None
                }
                line = json.dumps(tmp, ensure_ascii=False)
                f.write(line + "\n")
            return ""
        elif max_gen_len < 50:
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, "
                           "it is suggested to make the prompt shorter")
            self.fewer_than_50 += 1
            with open(self.args.failed_parse_file, "a") as f:
                tmp = {
                    "error_type": "Fewer_than_50",
                    "prompt": queries[0],
                    "response": None
                }
                line = json.dumps(tmp, ensure_ascii=False)
                f.write(line + "\n")
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, temperature=self.args.temperature, top_p=self.args.top_p,
            max_new_tokens=max_gen_len,
            num_return_sequences=1,
        )

        results = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(results)

        return results


def initial_generator(args):
    gen = (getattr(args, "generator", None) or "").strip()
    gen_l = gen.lower()
    if not gen:
        raise ValueError("--generator must be a non-empty string")

    if "llama-hf" in gen_l:
        return LLamaModelHF(args=args)
    if "llama" in gen_l:
        return LLamaModel(args=args)

    # OpenAI-compatible Chat Completions API.
    # This includes:
    #   - OpenAI names like gpt-3.5-turbo / gpt-4
    #   - Local servers exposing OpenAI-compatible APIs (e.g. Qwen2.5-* via OPENAI_API_BASE)
    openai_models = {
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-instruct",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "text-davinci-003",
    }
    if gen in openai_models or os.environ.get("OPENAI_API_BASE"):
        return OpenAIChatGPTModel(args=args)

    # Backward-compatible: internal ChatGPTModel for unknown gpt* strings when no API base is configured.
    if "gpt" in gen_l:
        return ChatGPTModel(args=args)

    raise ValueError(
        f"Unknown --generator={gen!r}. Set OPENAI_API_BASE for an OpenAI-compatible server, "
        "or use a supported generator (gpt-*, llama, llama-hf)."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##Generator
    parser.add_argument("--generator", type=str, default=None, help="generate model name")
    parser.add_argument("--openai", type=bool, default=False, help="if use openai api")
    parser.add_argument("--generator_file_path", type=str, default="./models/llama-2-7b-chat",
                        help="the path for llama2 chat, you can download from github or huggingface")
    parser.add_argument("--generator_tokenizer_path", type=str, default="./models/tokenizer.model",
                        help="the path for llama2 chat tokenizer, you can download from github or huggingface")
    parser.add_argument("--temperature", type=float, default=0.3, help="the temperature for inference")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p for inference")
    parser.add_argument("--top_k", type=int, default=40, help="top_k for inference")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="max sequence length")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="max generate length")
    parser.add_argument("--max_batch_size", type=int, default=4, help="max bench size")

    parser.add_argument("--thought_config_path", type=str, default="./prompts/asqa/thought_prompt.json",
                        help="the tree of thought prompt config file path")
    parser.add_argument("--evidence_fusion_config_path", type=str, default="./prompts/evidence_fusion.json",
                        help="the evidence fusion prompt config file path")
    parser.add_argument("--evidence_summary_config_path", type=str, default="./prompts/evidence_summary.json",
                        help="the evidence summary prompt config file path")
    parser.add_argument("--conflict_evidence_config_path", type=str, default="./prompts/conflict_evidence.json",
                        help="the conflict evidence prompt config file path")
    parser.add_argument("--missing_evidence_config_path", type=str, default="./prompts/missing_evidence.json",
                        help="the missing evidence prompt config file path")
    parser.add_argument("--conflict_fusion_config_path", type=str, default="./prompts/conflict_fusion.json",
                        help="the conflict fusion prompt config file path")
    parser.add_argument("--response_config_path", type=str, default="./prompts/hotpotqa/response_prompt.json",
                        help="the response prompt config file path")

    args = parser.parse_args()

    if args.openai:
        model = OpenAIChatGPTModel(args=args)
    elif "gpt" in args.generator:
        model = ChatGPTModel(args=args)
    elif "llama-hf" in args.generator:
        model = LLamaModelHF(args=args)
    elif "llama" in args.generator:
        model = LLamaModel(args=args)

    if args.openai:
        query = input("scanf your question:")
        response = model.get_response(system_prompt="", queries=query)
    if "gpt" in args.generator:
        while True:
            query = input("scanf your question:")
            body = model.get_body(system_prompt="You are a helpful assistant.", input=query)
            response = model.get_response(body=body)
    elif "llama-hf" in args.generator:
        while True:
            query = input("scanf your question:")
            model.get_response(queries=[query], system_prompt="")
    elif "llama" in args.generator:
        while True:
            query = input("scanf your question:")
            model.get_response(queries=[query], system_prompt="")

    # prompt = model.get_hotpotqa_response_prompt(shot=3, question="what is your name?", documents=["my name is gpt", "my name is jpli"])
    # prompt = model.get_missing_evidence_prompt(shot=3, question="what is your name?", history=["my name is gpt", "my name is jpli"])

