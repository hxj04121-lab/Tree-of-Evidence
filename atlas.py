# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
import time
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

EMBEDDINGS_DIM: int = 768

logger = logging.getLogger(__name__)
IGNORE_INDEX: int = -100
BERT_MAX_SEQ_LENGTH: int = 512


def encode_passages(batch, tokenizer, max_length):
    bsz = len(batch)
    n = max([len(example) for example in batch])
    batch = [example + [""] * (n - len(example)) for example in batch]
    batch = reduce(lambda a, b: a + b, batch)
    tokens = tokenizer(
        batch,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )
    tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
    return tokens


class Atlas(nn.Module):
    def __init__(self, opt, reader, retriever, reader_tokenizer, retriever_tokenizer):
        super(Atlas, self).__init__()

        self.reader = reader
        self.retriever = retriever
        self.reader_tokenizer = reader_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.opt = opt

        if reader_tokenizer is not None:
            self.READER_ALL_TOKENS = list(self.reader_tokenizer.vocab.values())

    def _get_fp16_retriever_copy(self):
        if hasattr(self.retriever, "module"):
            retriever_to_copy = self.retriever.module
        else:
            retriever_to_copy = self.retriever
        return copy.deepcopy(retriever_to_copy).half().eval()

    @torch.no_grad()
    def _retrieve(
        self,
        index,
        topk,
        query,
        query_ids_retriever,
        query_mask_retriever,
        iter_stats={},
    ):
        self.retriever.eval()
        if len(query) > 0:
            query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        else:
            query_emb = torch.empty((0, EMBEDDINGS_DIM)).cuda()

        search_start = time.time()

        passages, scores = index.search_knn(query_emb, topk)
        iter_stats["runtime/search"] = (time.time() - search_start, 1)

        return passages, scores, query_emb

    @torch.no_grad()
    def retrieve(self, *args, **kwargs):
        passages, scores = self._retrieve(*args, **kwargs)[:2]
        return passages, scores

    def retriever_tokenize(self, query):
        if self.retriever_tokenizer:
            query_enc = self.retriever_tokenizer(
                query,
                max_length=BERT_MAX_SEQ_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            query_enc = _to_cuda(query_enc)
        else:
            query_enc = None
        return _to_cuda(query_enc)


def _to_cuda(tok_dict):
    return {k: v.cuda() for k, v in tok_dict.items()}