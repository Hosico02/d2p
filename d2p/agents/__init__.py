"""d2p.agents — Analyzer / Planner / Executor.

Split into three modules in 2026-05-20 (the file was nearing 1k LOC and
mixing three independent concerns). The public API is unchanged: import
the classes and helper functions from `d2p.agents` exactly as before.
"""
from __future__ import annotations

from .analyzer import (
    ANALYZER_SYS,
    ANALYZER_USER_TMPL,
    Analyzer,
    _normalize_feature,
)
from .planner import (
    PLANNER_SYS,
    PLANNER_USER_TMPL,
    Planner,
    _compress_history,
)
from .executor import (
    EXECUTOR_SYS,
    EXECUTOR_USER_TMPL,
    Executor,
    PATCH_RETRY_SYS,
    parse_executor_output,
    # parser + safety-net helpers exported for unit tests
    _apply_post_check_to_result,
    _apply_search_replace,
    _extract_assertion_summary,
    _format_patch_retry_user,
    _fuzzy_locate,
    _guard_destructive_write,
    _post_write_syntax_check,
    _reindent_to,
    _strip_outer_fence,
    _with_line_numbers,
)

__all__ = [
    # agents
    "Analyzer", "Planner", "Executor",
    # prompts
    "ANALYZER_SYS", "ANALYZER_USER_TMPL",
    "PLANNER_SYS", "PLANNER_USER_TMPL",
    "EXECUTOR_SYS", "EXECUTOR_USER_TMPL",
    "PATCH_RETRY_SYS",
    # parser
    "parse_executor_output",
    # private helpers (re-exported for the unit-test suite)
    "_normalize_feature",
    "_compress_history",
    "_apply_post_check_to_result",
    "_apply_search_replace",
    "_extract_assertion_summary",
    "_format_patch_retry_user",
    "_fuzzy_locate",
    "_guard_destructive_write",
    "_post_write_syntax_check",
    "_reindent_to",
    "_strip_outer_fence",
    "_with_line_numbers",
]
