# Tree-of-Evidence

Official code for the paper: **Tree-of-Evidence: Iterative Evidence Retrieval for Log-based Anomaly Detection with Large Language Models**.

## Overview

Tree-of-Evidence (ToE) is an iterative evidence retrieval framework that enhances LLM-based log anomaly detection by constructing a tree of evidence through multi-round retrieval and reasoning. The system retrieves relevant log entries as evidence, reasons about their relevance, and iteratively refines queries to build comprehensive evidence for anomaly judgment.

Key components:
- **Tree-of-Thought Search**: Iterative retrieval with LLM-guided query refinement
- **Cross-block Evidence Fusion**: Aggregates evidence across log blocks

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA (recommended for dense retrieval)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py                         # Main entry point
├── evidence_tree/                  # Core Tree-of-Evidence search (package)
│   ├── __init__.py                 # Unified TreeOfEvidence class
│   ├── utils.py                    # Shared helpers and regex constants
│   ├── base.py                     # Base class (init, generate, logging)
│   ├── parsing.py                  # LLM response parsing
│   ├── fusion.py                   # Evidence fusion and conflict resolution
│   ├── core_tot.py                 # Basic tree-of-thought DFS retrieval
│   └── block_chain.py              # Cross-block retrieval and voting
├── searcher.py                     # Retrieval component (GTR / BM25 / Contriever)
├── generator.py                    # LLM generation component (OpenAI API)
├── index.py                        # FAISS index management
├── index_io.py                     # Index I/O utilities
├── utils.py                        # Text processing utilities
├── dist_utils.py                   # Distributed training utilities
├── slurm.py                        # SLURM cluster integration
├── atlas.py                        # Dense encoder (Contriever)
├── modeling_bert.py                # BERT model implementation
├── prompts.py                      # Prompt template generation
├── evaluate_aiops_anomaly_detection.py  # Anomaly classification metrics
├── evaluate_aiops_logs.py          # AIOps log evaluation (F1, EM)
├── evaluate_logs.py                # Log-match evaluation
├── evaluate/                       # Additional evaluation utilities
├── prompts/                        # Prompt configuration JSONs
└── scripts/                        # Experiment driver scripts
```

## Data Preparation

1. **HDFS**: Download the HDFS log dataset and prepare test queries in JSON format with fields: `question`, `answer`, `label`.
2. **BGL**: Download the BGL log dataset and prepare similarly.
3. **Thunderbird**: Download the Thunderbird log dataset.

Place the corpus JSONL files and pre-built FAISS indices under the `data/` directory.

## Usage

### Basic Inference

```bash
python main.py \
    --test_file_path data/hdfs_test_queries.json \
    --output_file_path result/hdfs_result.json \
    --corpus_path data/converted_logs/hdfs_corpus.jsonl \
    --index_path data/hdfs_index \
    --generator gpt-4o-mini \
    --retrieval_mode block_chain_tot \
    --thought_config_path prompts/thought_prompt_v1.json \
    --response_config_path prompts/response_prompt.json \
    --ndocs 5 \
    --depth 3
```

### Key Arguments

| Argument | Description |
|---|---|
| `--retrieval_mode` | Retrieval strategy: `tot`, `block_chain_tot`, `block_chain_vote`, etc. |
| `--generator` | LLM backend: `gpt-4o-mini`, `gpt-4o`, etc. |
| `--depth` | Maximum search depth for tree-of-thought |
| `--ndocs` | Number of documents to retrieve per round |
| `--thought_config_path` | Path to thought prompt template |
| `--response_config_path` | Path to response prompt template |
| `--evidence_fusion_strategy` | Evidence fusion method: `direct`, `vote`, `two_stage` |

### Experiment Scripts

The `scripts/` directory contains utility scripts for result analysis:

```bash
# Bootstrap confidence intervals for main table results
python scripts/bootstrap_ci.py --result_path result/hdfs_result.json --test_json data/hdfs_test.json

# Compile results across multiple experiment runs
python scripts/compile_all_results.py --result_dir result/

# Compute cost report (token usage, latency)
python scripts/compute_cost_report.py --result_path result/hdfs_result.json
```

### Evaluation

```bash
# Anomaly detection (classification metrics)
python evaluate_aiops_anomaly_detection.py --result_path result/hdfs_result.json

# Log evidence matching (F1, EM)
python evaluate_aiops_logs.py --hdfs_result result/hdfs_result.json
```

## Environment Variables

Set the following before running:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com"  # or your proxy
```

## License

This project is released under the MIT License.
