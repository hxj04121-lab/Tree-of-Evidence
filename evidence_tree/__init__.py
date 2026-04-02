"""
Tree-of-Evidence: Iterative Evidence Retrieval Framework.

This package implements the Tree-of-Evidence (ToE) search strategy for
log-based anomaly detection with LLMs. The monolithic module is split
into focused sub-modules for readability:

- ``utils``        – standalone helper functions and regex constants
- ``base``         – TreeOfEvidenceBase (init, generate, logging)
- ``parsing``      – LLM response parsing (thought / fusion / evidence)
- ``fusion``       – evidence fusion and conflict resolution
- ``core_tot``     – basic tree-of-thought DFS retrieval
- ``block_chain``  – cross-block retrieval and voting
"""

from evidence_tree.utils import (  # noqa: F401
    doc_to_text,
    _extract_log_target,
    _extract_doc_text_line,
    _select_extractive_log_response,
    _extract_thunderbird_program,
    _maybe_truncate_to_target_prefix,
)
from evidence_tree.base import TreeOfEvidenceBase
from evidence_tree.parsing import ParsingMixin
from evidence_tree.fusion import FusionMixin
from evidence_tree.core_tot import CoreToTMixin
from evidence_tree.block_chain import BlockChainMixin


class TreeOfEvidence(
    BlockChainMixin,
    CoreToTMixin,
    FusionMixin,
    ParsingMixin,
    TreeOfEvidenceBase,
):
    """Unified facade – inherits all retrieval modes via mixins."""
    pass
