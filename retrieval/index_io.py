# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging

import dist_utils
from retrieval.index import DistributedFAISSIndex, DistributedIndex

logger = logging.getLogger(__name__)


def load_passages(filenames, maxload=-1):
    def process_jsonl(
        fname,
        counter,
        passages,
        world_size,
        global_rank,
        maxload,
    ):
        def load_item(line):
            if line.strip() != "":
                item = json.loads(line)
                assert "id" in item
                if "title" in item and "section" in item and len(item["section"]) > 0:
                    item["title"] = f"{item['title']}: {item['section']}"
                return item
            else:
                print("empty line")

        for line in open(fname):
            if maxload > -1 and counter >= maxload:
                break

            ex = None
            if (counter % world_size) == global_rank:
                ex = load_item(line)
                passages.append(ex)
            counter += 1
        return passages, counter

    counter = 0
    passages = []
    global_rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    for filename in filenames:

        passages, counter = process_jsonl(
            filename,
            counter,
            passages,
            world_size,
            global_rank,
            maxload,
        )

    return passages


def save_embeddings_and_index(index, opt: argparse.Namespace) -> None:
    """
    Saves embeddings and passages files. It also saves faiss index files if FAISS mode is used.
    """
    index.save_index(opt.save_index_path, opt.save_index_n_shards)


def load_or_initialize_index(opt):
    index = DistributedIndex()
    passages = []

    if opt.load_index_path is not None:
        logger.info(f"Loading index from: {opt.load_index_path}")
        index.load_index(opt.load_index_path, opt.save_index_n_shards)
        passages = [index.doc_map[i] for i in range(len(index.doc_map))]
    else:
        logger.warning("You must pass the --load_index_path parameter to initialize retriever")
    return index, passages
