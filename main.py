import argparse
import json
import logging
import os
import warnings

import numpy as np

from generator import initial_generator
from searcher import RetrivalModel
from evidence_tree import TreeOfEvidence, _select_extractive_log_response
import dist_utils
import torch.distributed as dist


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    retriever = RetrivalModel(args)
    generator = initial_generator(args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=args)

    test_data = json.load(open(args.test_file_path))
    if args.quick_test_samples is not None:
        if not dist.is_initialized() or dist_utils.get_rank() == 0:
            # test_ids = np.random.choice(len(test_data), args.quick_test_samples, replace=False)
            # test_data = [test_data[int(idx)] for idx in test_ids]
            test_data = test_data[: args.quick_test_samples]
            if dist.is_initialized():
                dist.broadcast_object_list([test_data], src=0)
        else:
            data_list = [None]
            dist.broadcast_object_list(data_list, src=0)
            test_data = data_list[0]
    if not dist.is_initialized() or dist_utils.get_rank() == 0:
        if os.path.isfile(args.output_file_path):
            with open(args.output_file_path, "r") as f:
                result = json.load(f)
        else:
            result = []
        if dist.is_initialized():
            dist.broadcast_object_list([result], src=0)

        if args.reasoning_path_file is not None:
            if os.path.isfile(args.reasoning_path_file):
                with open(args.reasoning_path_file, "r") as f:
                    tree_result = json.load(f)
            else:
                tree_result = []
        else:
            tree_result = []
    else:
        result_list = [None]
        dist.broadcast_object_list(result_list, src=0)
        result = result_list[0]

    flush_every = int(getattr(args, "flush_every", 5) or 5)
    if flush_every < 1:
        flush_every = 1

    for cnt, line in enumerate(test_data):
        if cnt < len(result):
            continue
        query = line["question"]
        if world_size > 1:
            if getattr(args, "retrieval_mode", "tot") == "block_chain":
                logger.warning("block_chain mode currently runs in single-process mode; falling back to ToT(distributed).")
            if getattr(args, "retrieval_mode", "tot") == "block_chain_vote":
                logger.warning("block_chain_vote mode currently runs in single-process mode; falling back to ToT(distributed).")
            if args.with_model_evidence:
                response, reference, evidence, tree = tree_helper.tree_of_thought_missing_evidence_without_fusion_distribute(query=query)
            else:
                response, reference, evidence, tree = tree_helper.tree_of_thought_without_fusion_distribute(query=query)
        else:
            retrieval_mode = getattr(args, "retrieval_mode", "tot")
            if retrieval_mode == "block_chain":
                response, reference, evidence, tree = tree_helper.block_chain_retrieval(query=query)
            elif retrieval_mode == "block_chain_vote":
                response, reference, evidence, tree = tree_helper.block_chain_vote_retrieval(query=query)
            elif retrieval_mode == "block_chain_tot":
                response, reference, evidence, tree = tree_helper.tree_of_thought_with_cross_block(query=query)
            else:
                response, reference, evidence, tree = tree_helper.tree_of_thought_without_fusion(query=query)

        # Log-match convenience fields:
        # - response_full: best full log line found in documents (if any)
        # - response_prefix: the normalized query prefix (TARGET) for prefix queries (if any)
        response_full = _select_extractive_log_response(query, reference, response_mode="line")
        response_prefix = _select_extractive_log_response(query, reference, response_mode="prefix")
        result.append({
            "question": query,
            "evidence": evidence,
            "documents": reference,
            "response": response,
            "response_mode": getattr(args, "log_response_mode", "line"),
            "response_full": response_full,
            "response_prefix": response_prefix,
        })
        if dist_utils.get_rank() == 0:
            tree["idx"] = str(cnt)
            tree_result.append(tree)
            if cnt % flush_every == (flush_every - 1):
                with open(args.output_file_path, "w") as f:
                    json.dump(result, f)
                if args.reasoning_path_file is not None:
                    with open(args.reasoning_path_file, "w") as f:
                        json.dump(tree_result, f)
        dist_utils.barrier()

    # Flush the last partial batch (important for quick_test_samples not divisible by 5).
    if not dist.is_initialized() or dist_utils.get_rank() == 0:
        if args.output_file_path:
            with open(args.output_file_path, "w") as f:
                json.dump(result, f)
        if args.reasoning_path_file is not None:
            with open(args.reasoning_path_file, "w") as f:
                json.dump(tree_result, f)

    if not dist.is_initialized() or dist_utils.get_rank() == 0:
        logger.info("Total test data :{}".format(len(test_data)))
        total_thought = tree_helper.success_thought + tree_helper.failed_thought
        logger.info("Total thought times :{}".format(total_thought))
        logger.info("Success thought times :{}".format(tree_helper.success_thought))
        logger.info("Failed thought times :{}".format(tree_helper.failed_thought))
        if total_thought > 0:
            logger.info("Success ratio :{:.2%}".format(tree_helper.success_thought / total_thought))
            logger.info("Failed ratio :{:.2%}".format(tree_helper.failed_thought / total_thought))
        else:
            logger.info("Success ratio :N/A (no thoughts generated)")
            logger.info("Failed ratio :N/A (no thoughts generated)")
        logger.info("Total call generator times:{}".format(generator.called_times))
        logger.info("Input sequence over max seq length times:{}".format(generator.prompt_exceed_max_length))
        logger.info("Generate sequence length fewer than 50 times:{}".format(generator.fewer_than_50))
    return


def post_evidence_fusion(args):
    retriever = None
    generator = initial_generator(args)
    tree_helper = TreeOfEvidence(retriever=retriever, generator=generator, args=args)

    test_data = json.load(open(args.test_file_path))

    if os.path.isfile(args.output_file_path):
        with open(args.output_file_path, "r") as f:
            result = json.load(f)
    else:
        result = []

    for cnt, line in enumerate(test_data):
        if cnt < len(result):
            continue
        if args.quick_test_samples is not None and cnt >= args.quick_test_samples:
            break
        line["idx"] = str(cnt)
        categories = tree_helper.post_evidence_fusion(line)
        if categories is None:
            continue
        line["categories"] = categories
        result.append(line)
        if cnt % 5 == 4:
            with open(args.output_file_path, "w") as f:
                json.dump(result, f)

    with open(args.output_file_path, "w") as f:
        json.dump(result, f)
    return


if __name__=="__main__":
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
    parser.add_argument("--per_gpu_embedder_batch_size", default=512, type=int, help="Embedder's batch size per GPU.")
    parser.add_argument("--local-rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node jobs)")

    ##Generator
    parser.add_argument("--generator", type=str, default=None, help="generate model name")
    parser.add_argument("--generator_file_path", type=str, default="./models/llama-2-7b-chat",
                        help="the path for llama2 chat, you can download from github or huggingface")
    parser.add_argument("--generator_tokenizer_path", type=str, default="./models/tokenizer.model",
                        help="the path for llama2 chat tokenizer, you can download from github or huggingface")
    parser.add_argument("--temperature", type=float, default=0., help="the temperature for inference")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p for inference")
    parser.add_argument("--top_k", type=int, default=40, help="top_k for inference")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="max sequence length")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="max generate length")
    parser.add_argument(
        "--prompt_doc_max_chars",
        type=int,
        default=0,
        help="Optional: truncate each document in response prompt to N chars (0=off).",
    )
    parser.add_argument(
        "--prompt_query_max_chars",
        type=int,
        default=0,
        help="Optional: truncate query in response prompt to N chars (0=off).",
    )
    parser.add_argument("--max_batch_size", type=int, default=4, help="max bench size")
    parser.add_argument(
        "--log_response_mode",
        type=str,
        default="line",
        choices=["line", "prefix", "auto"],
        help=(
            "How to format the final log-match response. "
            "line: return full matched log line. "
            "prefix: for prefix queries (ending with '...'), return only TARGET (question substring after "
            "'Find logs with message:' with trailing '...' removed). "
            "auto: return prefix only for syslog-style prefix queries; otherwise behave like line."
        ),
    )

    # Log-match / retrieval fairness toggles (for ablations)
    parser.add_argument(
        "--enable_log_match_fastpath",
        type=int,
        default=1,
        help="1=enable extractive fastpath shortcuts for log-match; 0=disable (closer to original ToT flow).",
    )
    parser.add_argument(
        "--normalize_log_query",
        type=int,
        default=1,
        help="1=normalize 'Find logs with message: ...' queries for dense retrieval; 0=keep raw queries (closer to original).",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=5,
        help="Write partial JSON outputs every N samples (default: 5).",
    )
    parser.add_argument(
        "--use_extractive_response",
        type=int,
        default=1,
        help=(
            "1=prefer extractive (document-based) responses when the exact/prefix match exists; "
            "0=always let LLM generate the final response (closer to the original ToT behavior)."
        ),
    )
    parser.add_argument(
        "--log_match_skip_thought",
        type=int,
        default=0,
        help=(
            "For log-match queries only: 1=skip ToT thought/DFS and answer purely by retrieval + extractive match; "
            "0=run full ToT (default). Recommended for fast, clean cross-block ablations when combined with "
            "--enable_log_match_fastpath 0."
        ),
    )
    parser.add_argument(
        "--block_chain_prefix_output",
        type=int,
        default=0,
        help=(
            "For log-match prefix queries (works in tot and block_chain_tot): control optional truncation.\n"
            "  0=off (keep full line),\n"
            "  1=always truncate to TARGET when grounded,\n"
            "  2=smart truncate only when grounded and (len(response)-len(TARGET)) >= --block_chain_prefix_output_extra_ge.\n"
        ),
    )
    parser.add_argument(
        "--block_chain_prefix_output_extra_ge",
        type=int,
        default=50,
        help=(
            "Used only when --block_chain_prefix_output=2: truncate to TARGET only if the grounded full-line "
            "copy contains at least this many extra characters beyond TARGET (default: 50)."
        ),
    )

    ##Tree of Thought
    parser.add_argument("--max_depth", type=int, default=3, help="max search depth for evidence tree")
    parser.add_argument("--max_nodes", type=int, nargs="+", help="max search nodes for each level")
    parser.add_argument("--with_model_evidence", type=bool, default=True, help="model generate query or evidence")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="tot",
        choices=["tot", "block_chain", "block_chain_tot", "block_chain_vote"],
        help=(
            "Retrieval strategy: tot (default), block_chain (HDFS-only multi-block chained retrieval, no ToT), "
            "block_chain_tot (cross-block retrieval + ToT reasoning on supported log corpora), "
            "or block_chain_vote (cross-block retrieval + per-block candidates + LLM vote, no ToT DFS)."
        ),
    )
    parser.add_argument(
        "--block_chain_block_map_path",
        type=str,
        default=None,
        help=(
            "Optional TSV mapping file for block keys (native_key -> super_block_key). "
            "When set, the retriever builds the block index using mapped keys (useful for random/KMeans super-block ablations)."
        ),
    )
    parser.add_argument(
        "--block_chain_block_map_kind",
        type=str,
        default="native",
        choices=["native", "host_superblock"],
        help=(
            "Interpretation of keys in --block_chain_block_map_path. "
            "native: map the dataset's native block key (HDFS=blk_*, BGL=title, Thunderbird=host). "
            "host_superblock: mapping is keyed by Thunderbird host (first token of syslog line)."
        ),
    )
    parser.add_argument(
        "--block_chain_other_blocks",
        type=int,
        default=9,
        help="How many other blocks to consider (besides the query block)",
    )
    parser.add_argument("--block_chain_pool", type=int, default=2000, help="Global candidate pool size for selecting other blocks")
    parser.add_argument("--block_chain_min_similarity", type=float, default=0.35, help="Min cosine similarity to fuse cross-block evidence")
    parser.add_argument("--block_chain_early_stop_no_new", type=int, default=1, help="Stop if no new evidence added for N consecutive depths")
    parser.add_argument(
        "--block_chain_cross_top_k",
        type=int,
        default=1,
        help=(
            "For block_chain_tot mode only: retrieve top-K candidates per other block, "
            "then select the best per block before choosing the global best cross-block evidence."
        ),
    )
    parser.add_argument(
        "--block_chain_cross_append_mode",
        type=str,
        default="best",
        choices=["best", "all"],
        help=(
            "For block_chain_tot mode only: how to add cross-block documents to the per-depth candidate list.\n"
            "  best: append only the single best cross-block document across other blocks (default; less noise)\n"
            "  all : append the best document from each other block (up to --block_chain_other_blocks); "
            "useful for ablations where you want to measure cross-block contributions explicitly."
        ),
    )
    parser.add_argument(
        "--block_chain_skip_cross_if_self_score_ge",
        type=float,
        default=1.0,
        help=(
            "For block_chain_tot mode only: if the best self-block retrieval score is >= this threshold, "
            "skip cross-block fusion (helps precision by avoiding noisy evidence)."
        ),
    )
    parser.add_argument(
        "--block_chain_cross_program_strict",
        type=int,
        default=0,
        help=(
            "For Thunderbird block_chain_tot: if enabled, require cross-block candidates to have the same program "
            "as TARGET before per-block selection."
        ),
    )
    parser.add_argument(
        "--block_chain_thunderbird_self_block_mode",
        type=str,
        default="query",
        choices=["query", "shift"],
        help=(
            "For Thunderbird block_chain_tot/block_chain_vote only: how to determine the self block (host).\n"
            "  query: use the host token from TARGET (default; normal behavior)\n"
            "  shift: intentionally map host->next host in sorted host list (deterministic); "
            "useful for hard ablations proving cross-block recovery when self-block is wrong."
        ),
    )
    parser.add_argument(
        "--block_chain_self_block_mode",
        type=str,
        default="normal",
        choices=["normal", "shift"],
        help=(
            "For block_chain_tot/block_chain_vote: optionally force the self-block to a wrong block (shift) "
            "for hard ablations proving cross-block recovery. "
            "normal: use the inferred/query-derived self-block (default). "
            "shift: map self_block -> next block key in the built block index (deterministic)."
        ),
    )

    parser.add_argument(
        "--hdfs_block_prefix_scan",
        type=int,
        default=1,
        help=(
            "For HDFS block_chain_tot prefix queries only: when blk_... in the query is truncated (key not found in "
            "block index), scan candidate blocks by prefix and lexically match TARGET inside those blocks. "
            "0=disable (more realistic, but may increase missing), 1=enable."
        ),
    )
    parser.add_argument(
        "--block_chain_vote_top_k_self",
        type=int,
        default=None,
        help="For block_chain_vote only: top-K docs retrieved from self block (default: --top_k_documents).",
    )
    parser.add_argument(
        "--block_chain_vote_top_k_other",
        type=int,
        default=None,
        help="For block_chain_vote only: top-K docs retrieved per other block (default: --block_chain_cross_top_k).",
    )
    parser.add_argument(
        "--block_chain_vote_judge_model",
        type=str,
        default=None,
        help="Optional: use a separate model for voting/judging (default: same as --generator).",
    )
    parser.add_argument(
        "--block_chain_vote_judge_prompt_path",
        type=str,
        default="prompts/logs_vote_prompt_v1.json",
        help="Prompt config JSON for block_chain_vote judge (default: prompts/logs_vote_prompt_v1.json).",
    )

    parser.add_argument(
        "--gate_mode",
        type=str,
        default="full",
        choices=["full", "vr_only", "no_gate"],
        help=(
            "For ToT-style thought parsing: gate ablations.\n"
            "full=use relevance+support gates; vr_only=only relevance gate; no_gate=disable gates (trust model Step 4 / Step 3)."
        ),
    )
    parser.add_argument(
        "--evidence_fusion_strategy",
        type=str,
        default="direct",
        choices=["direct", "vote", "two_stage"],
        help=(
            "Evidence fusion strategy. "
            "direct=concatenate all docs and call LLM once (default); "
            "vote=per-block judgments + majority vote; "
            "two_stage=per-block summaries then a final global judgment."
        ),
    )


    ##Prompts
    parser.add_argument("--thought_shot", type=int, default=1, help="few shot for tree of thought prompt")
    parser.add_argument("--evidence_fusion_shot", type=int, default=1, help="few shot for evidence fusion prompt")
    parser.add_argument("--evidence_summary_shot", type=int, default=1, help="few shot for evidence summary prompt")
    parser.add_argument("--conflict_evidence_shot", type=int, default=1,
                        help="few shot for the prompt that asks question about conflicting evidence")
    parser.add_argument("--missing_evidence_shot", type=int, default=1,
                        help="few shot for the prompt that asks question about missing evidence")
    parser.add_argument("--conflict_fusion_shot", type=int, default=1,
                        help="few shot for the prompt that decide which evidence to accept")
    parser.add_argument("--response_shot", type=int, default=1,
                        help="few shot for the response prompt")

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
    parser.add_argument("--response_config_path", type=str, default="./prompts/response.json",
                        help="the response prompt config file path")

    ## Others
    parser.add_argument("--failed_parse_file", type=str, help="Path for record failed parse the tot response")
    parser.add_argument("--reasoning_path_file", type=str, help="Path for record reasoning path")

    ## test
    parser.add_argument("--test_file_path", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test_samples", type=int, help="Quickly test a few examples")
    parser.add_argument("--output_file_path", type=str, help="Path to dump eval file")

    args = parser.parse_args()

    main(args)
    # post_evidence_fusion(args)
