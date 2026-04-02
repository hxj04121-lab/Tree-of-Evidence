import csv
import json
import logging
import os
import pickle
import re
import time as _time_mod
import warnings
from array import array

from sentence_transformers import SentenceTransformer
from index_io import load_or_initialize_index

from atlas import Atlas
from modeling_bert import BertModel
from tqdm import tqdm
import argparse
import torch
import dist_utils
import torch.distributed as dist
import slurm

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HDFS_BLOCK_RE = re.compile(r"blk_-?\d+")
_NO_BLOCK_KEY = "__NO_BLOCK__"
_LOG_QUERY_PREFIX_RE = re.compile(r"(?i)^\s*(?:\\[query\\]\\s*)?find\\s+logs\\s+with\\s+message\\s*:\\s*")


def _normalize_log_match_query(q: str) -> str:
    """
    For log-match datasets, queries are formatted like:
      "Find logs with message: <syslog line prefix>..."
    Embedding the instruction text and trailing ellipsis hurts retrieval quality, so we strip them.

    This keeps the *original* question for downstream prompts/evaluation, but uses a cleaner string
    for dense retrieval.
    """
    raw = (q or "").strip()
    if not _LOG_QUERY_PREFIX_RE.search(raw):
        return q

    is_prefix_query = raw.rstrip().endswith("...")
    target = _LOG_QUERY_PREFIX_RE.sub("", raw).strip()
    if is_prefix_query and target.endswith("..."):
        target = target[:-3].strip()
    target = " ".join(target.split())
    if target and target[0] in {"\"", "'"}:
        target = target[1:].lstrip()
    if target and target[-1] in {"\"", "'"}:
        target = target[:-1].rstrip()
    return target or q


def _infer_log_dataset_from_path(wiki_passage: str):
    name = os.path.basename(wiki_passage or "").lower()
    if "hdfs" in name:
        return "hdfs"
    if "bgl" in name:
        return "bgl"
    if "thunderbird" in name or "tbird" in name:
        return "thunderbird"
    return None


def doc_to_text_tfidf(doc):
    return doc['title'] + ' ' + doc['text']


def doc_to_text_dense(doc):
    return doc['title'] + '. ' + doc['text']


class RetrivalModel:
    def __init__(self, args):
        slurm.init_distributed_mode(args)
        self.args = args
        self.docs = []
        self.embedding = None
        self.block_to_idx = None
        self.block_offsets = None
        self.block_doc_indices = None
        self.block_index_enabled = False
        if "gtr" in self.args.retriever:
            self.model = self.get_gtr_retriever()
        elif "bm25" in self.args.retriever:
            self.model = self.get_bm25_retriever()
        elif "contriever" in self.args.retriever:
            self.model, self.tokenizer = self.get_atlas_retriever()
        else:
            logger.warning("To load correct retriever, you must set --retriever with bm25/gtr")
        return

    def get_gtr_retriever(self):
        logger.info("*" * 20 + "Start loading GTR retriever" + "*" * 20)
        logger.info("loading GTR encoder...")
        device = self.args.retriever_device if torch.cuda.is_available() else "cpu"
        encoder = SentenceTransformer(self.args.retriever_model_name_or_path, device=device,
                                      cache_folder=self.args.retriever_cache)

        logger.info("loading wikipedia file...")
        if self.args.wiki_passage is None:
            logger.warning("You must pass the --wiki_passage parameter to initialize retriever")
        enable_block_index = getattr(self.args, "retrieval_mode", "tot") in ("block_chain", "block_chain_tot", "block_chain_vote", "aiops_block_chain_tot", "aiops_chimera_cder", "aiops_chimera_cder_tot", "aiops_chimera_dualview")
        log_dataset = _infer_log_dataset_from_path(self.args.wiki_passage) if self.args.wiki_passage else None
        build_block_index = enable_block_index and log_dataset in ("hdfs", "bgl", "thunderbird")
        block_key_map = None
        block_key_map_path = getattr(self.args, "block_chain_block_map_path", None)
        if build_block_index and block_key_map_path:
            try:
                block_key_map = {}
                with open(block_key_map_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    for i, row in enumerate(reader):
                        if not row or len(row) < 2:
                            continue
                        k = str(row[0] or "").strip()
                        v = str(row[1] or "").strip()
                        if not k or not v:
                            continue
                        if i == 0 and k.lower() in {"native_key", "host", "key", "from"}:
                            continue
                        block_key_map[k.lower()] = v.lower()
                if not block_key_map:
                    logger.warning("block map file provided but empty: %s", block_key_map_path)
                    block_key_map = None
                else:
                    logger.info("Loaded block key map: %s (entries=%d)", block_key_map_path, len(block_key_map))
            except Exception as e:
                logger.warning("Failed to load block key map %s: %s", block_key_map_path, e)
                block_key_map = None
        if build_block_index:
            logger.info("building %s block index (for %s mode)...", log_dataset.upper(), getattr(self.args, "retrieval_mode", "tot"))
            block_to_idx = {}
            block_counts = []
            doc_block_idx = array("I")

        with open(self.args.wiki_passage, encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                # The converted log corpora are TSV-like, but may occasionally contain malformed
                # rows (e.g., missing the trailing "title" column on the last line). Keep the
                # corpus/embedding alignment by treating missing fields as empty strings.
                if not row:
                    text = ""
                    title = ""
                elif len(row) == 1:
                    text = ""
                    title = ""
                elif len(row) == 2:
                    text = row[1]
                    title = ""
                elif len(row) == 3:
                    text = row[1]
                    title = row[2]
                else:
                    # Schema: id \t text \t title. If text ever contains tabs, join all middle fields.
                    text = "\t".join(row[1:-1])
                    title = row[-1]
                self.docs.append(title + "\n" + text)
                if build_block_index:
                    if log_dataset == "hdfs":
                        m = _HDFS_BLOCK_RE.search(text)
                        block_id = m.group(0) if m else _NO_BLOCK_KEY
                    elif log_dataset == "bgl":
                        block_id = title or _NO_BLOCK_KEY
                    else:
                        # Thunderbird: block key is host (first token of the syslog line).
                        t = (text or "").strip()
                        block_id = (t.split(None, 1)[0] if t else "") or _NO_BLOCK_KEY

                    block_key = str(block_id).strip().lower()
                    if block_key_map is not None:
                        block_key = block_key_map.get(block_key, block_key)
                    idx = block_to_idx.get(block_key)
                    if idx is None:
                        idx = len(block_to_idx)
                        block_to_idx[block_key] = idx
                        block_counts.append(0)
                    block_counts[idx] += 1
                    doc_block_idx.append(idx)

        if build_block_index:
            offsets = [0]
            total = 0
            for c in block_counts:
                total += c
                offsets.append(total)

            doc_indices = array("I", [0]) * len(doc_block_idx)
            pos = offsets[:-1].copy()
            for doc_idx, blk_idx in enumerate(doc_block_idx):
                p = pos[blk_idx]
                doc_indices[p] = doc_idx
                pos[blk_idx] = p + 1

            self.block_to_idx = block_to_idx
            self.block_offsets = offsets
            self.block_doc_indices = doc_indices
            self.doc_block_idx = doc_block_idx
            self.block_index_enabled = True
            self.block_key_mode = log_dataset
            self.block_map_enabled = bool(block_key_map)
            self.block_key_map_path = block_key_map_path if block_key_map_path else None
            idx_to_block = [None] * len(block_to_idx)
            for key, blk_idx in block_to_idx.items():
                idx_to_block[blk_idx] = key
            self.block_idx_to_key = idx_to_block
            logger.info(
                "%s block index ready: blocks=%d docs=%d",
                log_dataset.upper(),
                len(block_to_idx),
                len(doc_block_idx),
            )

        logger.info("loading gtr wikipedia index...")
        if not os.path.exists(self.args.gtr_embedding):
            logger.info("gtr embeddings not found, building... This will cost a few minutes.")
            embedding = self.gtr_build_index(encoder, self.docs)
        else:
            logger.info("gtr embeddings found, loading...")
            with open(self.args.gtr_embedding, "rb") as f:
                embedding = pickle.load(f)
        self.embedding = torch.tensor(embedding, dtype=torch.float16, device=self.args.embedding_device)
        return encoder

    def get_gtr_documents_in_blocks(self, questions, block_ids, top_k=None):
        if not self.block_index_enabled:
            raise RuntimeError(
                "Block index not initialized. Use --retrieval_mode block_chain/block_chain_tot on a supported log corpus."
            )
        if len(questions) != len(block_ids):
            raise ValueError("questions and block_ids must have the same length")

        requested_k = self.args.top_k_documents if top_k is None else top_k
        result = []
        device = self.args.embedding_device if torch.cuda.is_available() else "cpu"

        do_norm = int(getattr(self.args, "normalize_log_query", 1))
        norm_questions = [_normalize_log_match_query(q) for q in questions] if do_norm else list(questions)
        _t0 = _time_mod.time()
        with torch.inference_mode():
            queries = self.model.encode(
                norm_questions, batch_size=4, show_progress_bar=False, normalize_embeddings=True
            )
            queries = torch.tensor(queries, dtype=torch.float16, device="cpu")

        for qi, q in enumerate(queries):
            q = q.to(device)
            block_id = str(block_ids[qi]).strip().lower()
            blk_idx = self.block_to_idx.get(block_id) if self.block_to_idx is not None else None
            if blk_idx is None or self.block_offsets is None or self.block_doc_indices is None:
                docs = []
            else:
                start = self.block_offsets[blk_idx]
                end = self.block_offsets[blk_idx + 1]
                if end <= start:
                    docs = []
                else:
                    cand = self.block_doc_indices[start:end]
                    cand_idx = torch.tensor(cand, dtype=torch.long, device=device)
                    cand_emb = self.embedding.index_select(0, cand_idx)
                    scores = torch.matmul(cand_emb, q)
                    k = min(requested_k, scores.size(0))
                    score, idx_local = torch.topk(scores, k)
                    docs = []
                    for i in range(k):
                        global_idx = cand_idx[idx_local[i]].item()
                        title, text = self.docs[global_idx].split("\n", 1)
                        docs.append(
                            {
                                "id": str(global_idx + 1),
                                "title": title,
                                "text": text,
                                "score": score[i].item(),
                            }
                        )

            result.append({"question": questions[qi], "documents": docs})

        self.last_retrieval_latency_ms = (_time_mod.time() - _t0) * 1000
        return result

    def gtr_build_index(self, encoder, docs):
        with torch.inference_mode():
            embedding = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
            embedding = embedding.astype("float16")

        with open(self.args.gtr_embedding, "wb") as f:
            pickle.dump(embedding, f)
        return embedding

    def get_bm25_retriever(self):
        logger.info("*" * 20 + "Start loading BM25 retriever" + "*" * 20)
        logger.info("loading bm25 index, this may take a while...")
        if self.args.bm25_sphere_index is None:
            logger.warning("You must pass the --bm25_sphere_index parameter to initialize retriever")
        #encoder = LuceneSearcher(self.args.bm25_sphere_index)
        encoder = SentenceTransformer()
        return encoder

    def get_atlas_retriever(self):
        def _load_atlas_model_state(args, model, model_dict):
            model_dict = {
                k.replace("retriever.module", "retriever").replace("reader.module", "reader"): v for k, v in
                model_dict.items()
            }
            model_dict = {k: v for k, v in model_dict.items() if not k.startswith("reader")}
            model.load_state_dict(model_dict)
            model = model.to("cuda:{}".format(dist_utils.get_rank()))
            return model

        logger.info("loading contriever wikipedia index...")
        self.embedding, self.docs = load_or_initialize_index(self.args)
        # print("*"*30 + "sleep" + "*"*30)
        # time.sleep(10000)
        epoch_path = os.path.realpath(self.args.retriever_model_name_or_path)
        save_path = os.path.join(epoch_path, "model.pth.tar")
        logger.info(f"Loading {epoch_path}")
        logger.info(f"loading checkpoint {save_path}")
        checkpoint = torch.load(save_path, map_location="cpu")
        model_dict = checkpoint["model"]
        encoder = Contriever.from_pretrained(self.args.retriever_type)
        tokenizer = AutoTokenizer.from_pretrained(self.args.retriever_type)
        retriever = DualEncoderRetriever(self.args, encoder)
        retriever = Atlas(self.args, None, retriever, None, tokenizer)
        retriever = _load_atlas_model_state(self.args, retriever, model_dict)

        # logger.info("loading contriever wikipedia index...")
        # self.embedding, self.docs = load_or_initialize_index(self.args)

        return retriever, tokenizer

    def get_gtr_documents(self, questions, top_k=None):
        result = []
        device = self.args.embedding_device if torch.cuda.is_available() else "cpu"
        do_norm = int(getattr(self.args, "normalize_log_query", 1))
        norm_questions = [_normalize_log_match_query(q) for q in questions] if do_norm else list(questions)
        _t0 = _time_mod.time()
        with torch.inference_mode():
            queries = self.model.encode(norm_questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
            queries = torch.tensor(queries, dtype=torch.float16, device="cpu")
        for qi, q in enumerate(tqdm(queries)):
            q = q.to(device)
            scores = torch.matmul(self.embedding, q)
            score, idx = torch.topk(scores, self.args.top_k_documents if top_k is None else top_k)
            ret = []
            for i in range(idx.size(0)):
                title, text = self.docs[idx[i].item()].split("\n", 1)
                ret.append({"id": str(idx[i].item() + 1), "title": title, "text": text, "score": score[i].item()})
            result.append({
                "question": questions[qi],
                "documents": ret,
            })
        self.last_retrieval_latency_ms = (_time_mod.time() - _t0) * 1000
        return result

    def get_bm25_documents(self, question, top_k=None):
        result = []
        for query in tqdm(question):
            try:
                hits = self.model.search(query, self.args.top_k_documents if top_k is None else top_k)
            except Exception as e:
                # https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
                if "maxClauseCount" in str(e):
                    query = " ".join(query.split())[:950]
                    hits = self.model.search(query, self.args.top_k_documents if top_k is None else top_k)
                else:
                    raise e
            docs = []
            for hit in hits:
                h = json.loads(str(hit.docid).strip())
                docs.append({
                    "title": h["title"],
                    "text": hit.raw,
                    "url": h["url"],
                })
            result.append({
                "question": query,
                "documents": docs,
            })
        return result

    def get_atlas_documents(self, question, top_k=None):
        def get_unwrapped_model_if_wrapped(model):
            if hasattr(model, "module"):
                return model.module
            return model

        query_enc = self.model.retriever_tokenize(question)
        unwrapped_model = get_unwrapped_model_if_wrapped(self.model)
        retrieved_passages, score = unwrapped_model.retrieve(
            self.embedding,
            self.args.top_k_documents if top_k is None else top_k,
            question,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
        )
        result = []
        for idx, query in enumerate(question):
            documents = retrieved_passages[idx]
            result.append({
                "question": query,
                "documents": documents,
            })
        return result

    def get_documents(self, question, top_k=None):
        if "gtr" in self.args.retriever:
            return self.get_gtr_documents(question, top_k)
        if "bm25" in self.args.retriever:
            return self.get_bm25_documents(question, top_k)
        if "contriever" in self.args.retriever:
            return self.get_atlas_documents(question, top_k)
        logger.warning("To get correct documents with the retriever, you must set --retriever with bm25/gtr")


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0).clone()
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1).clone() / attention_mask.sum(dim=1)[..., None].clone()
        elif self.config.pooling == "sqrt":
            emb = last_hidden.sum(dim=1) / torch.sqrt(attention_mask.sum(dim=1)[..., None].float())
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1).clone()
        return emb


class BaseRetriever(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(BaseRetriever, self).__init__()

    def embed_queries(self, *args, **kwargs):
        raise NotImplementedError()

    def embed_passages(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, is_passages=False, **kwargs):
        if is_passages:
            return self.embed_passages(*args, **kwargs)
        else:
            return self.embed_queries(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        for m in self.children():
            m.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for m in self.children():
            m.gradient_checkpointing_disable()


class DualEncoderRetriever(BaseRetriever):
    def __init__(self, opt, contriever):
        super(DualEncoderRetriever, self).__init__()
        self.opt = opt
        self.contriever = contriever

    def _embed(self, *args, **kwargs):
        return self.contriever(*args, **kwargs)

    def embed_queries(self, *args, **kwargs):
        return self._embed(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        return self._embed(*args, **kwargs)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    ### Retriever
    parser.add_argument("--retriever", type=str, default=None, help="options: bm25/gtr/contriever")
    parser.add_argument("--retriever_model_name_or_path", type=str, default="sentence-transformers/gtr-t5-xxl",
                        help="the name or path for gtr retriever, you can pass a model name to download model "
                             "parameters from huggingface, or a local path to load the model locally")
    parser.add_argument("--retriever_type", type=str, default="bert-base-uncased",
                        help="")
    parser.add_argument("--retriever_cache", type=str, default="./models",
                        help="his parameter allows you to specify the cache path for your model "
                             "when you download it from huggingface")
    parser.add_argument("--retriever_device", type=str, default="cuda:0",
                        help="cuda device for loading model")
    parser.add_argument("--embedding_device", type=str, default="cuda:1",
                        help="cuda device for wiki embedding")

    ### Document file path
    parser.add_argument("--gtr_embedding", type=str, default=None, help="path of gtr wiki embedding")
    parser.add_argument("--wiki_passage", type=str, default=None, help="path of wiki passages")
    parser.add_argument("--bm25_sphere_index", type=str, default=None, help="path of bm25 index about sphere")
    parser.add_argument("--load_index_path", type=str, default=None, help="path of contriver index file path")
    parser.add_argument("--save_index_n_shards", type=int, default=128,
                        help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.")

    ### Retrieval prams
    parser.add_argument("--top_k_documents", type=int, default=100, help="top k documents will be return")

    ### Distribution prams
    parser.add_argument("--per_gpu_embedder_batch_size", default=8, type=int, help="Embedder's batch size per GPU.")
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node jobs)")

    args = parser.parse_args()
    retriever = RetrivalModel(args)

    # result = []
    # while True:
    #     if dist_utils.get_rank() == 0:
    #         question = input("please enter the question：")
    #         dist.broadcast_object_list([question], src=0)
    #     else:
    #         question_list = [None]
    #         dist.broadcast_object_list(question_list, src=0)
    #         question = question_list[0]
    #     if question == "exit":
    #         break
    #     response = retriever.get_documents(question=[question], top_k=5)
    #     result.append(response[0])
    # if dist_utils.get_rank() == 0:
    #     with open("./prompts/baselines/demo_documents_react_musique.json", "w") as f:
    #         json.dump(result, f)
    # while True:
    #
    #     question = input("please enter the question：")
    #     response = retriever.get_documents(question=[question], top_k=5)
    #     if dist_utils.get_rank() == 0:
    #         print(response)
    #         print("*" * 50)

    with open("./data/musique/ans_dev_first_500.json", "r") as f:
        data = json.load(f)
    question = [d["question"] for d in data]

    response = []
    for q in question:
        sub_response = retriever.get_documents(question=[q], top_k=15)
        response.extend(sub_response)
    if dist_utils.get_rank() == 0:
        with open("./result/baselines/musique/result/rag_retrieval.json", "w") as f:
            json.dump(response, f)

