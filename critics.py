"""The reflection critics — Self-RAG's ``IsRel`` / ``IsSup`` / ``IsUse`` and the
retrieve decision — plus the critic-weighted score used to pick the best answer.

Two layers, kept apart on purpose:

* **Pure parsers** (``parse_*``) turn a model's free text into a single discrete
  label. No network, fully deterministic, exhaustively unit-tested. They are
  deliberately forgiving — they read the *last* mentioned label and tolerate
  explanations, casing, and minor wording — and they fail *safe*: an unreadable
  relevance verdict is treated as IRRELEVANT, unreadable support as NO_SUPPORT,
  so ambiguity pushes the pipeline toward abstaining rather than bluffing.
* **Critic calls** (``need_retrieval``, ``is_relevant``, …) issue one ``chat_fn``
  call and parse the result. ``chat_fn`` is injected so tests run offline.
"""

from __future__ import annotations

import re

from .llm import chat
from .prompts import (
    RELEVANCE_PROMPT,
    RETRIEVE_DECISION_PROMPT,
    SUPPORT_PROMPT,
    USEFUL_PROMPT,
)

# IsSup labels and the grounding weight each one carries.
SUPPORT_LEVELS = ("FULLY", "PARTIALLY", "NO_SUPPORT")
_SUPPORT_WEIGHT = {"FULLY": 1.0, "PARTIALLY": 0.5, "NO_SUPPORT": 0.0}


def _scan(text: str, mapping: list[tuple[str, object]], default: object) -> object:
    """Return the value for the first needle found, scanning the last line first.

    ``mapping`` is checked in order within each line, so list more specific
    needles first (e.g. ``IRRELEVANT`` before ``RELEVANT``).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        up = line.upper()
        for needle, value in mapping:
            if needle in up:
                return value
    up = text.upper()
    for needle, value in mapping:
        if needle in up:
            return value
    return default


# --- pure parsers ---------------------------------------------------------

def parse_retrieve(text: str) -> bool:
    """True if retrieval is called for. Defaults to True (retrieve when unsure)."""
    return _scan(
        text,
        [("NO_RETRIEVE", False), ("NO RETRIEVE", False), ("RETRIEVE", True)],
        default=True,
    )


def parse_relevance(text: str) -> bool:
    """True if the passage is relevant. Defaults to False (fail safe: drop it)."""
    return _scan(
        text,
        [("IRRELEVANT", False), ("NOT RELEVANT", False), ("RELEVANT", True)],
        default=False,
    )


def parse_support(text: str) -> str:
    """One of SUPPORT_LEVELS. Defaults to NO_SUPPORT (fail safe)."""
    return _scan(
        text,
        [
            ("NO_SUPPORT", "NO_SUPPORT"),
            ("NO SUPPORT", "NO_SUPPORT"),
            ("NOT SUPPORT", "NO_SUPPORT"),
            ("UNSUPPORTED", "NO_SUPPORT"),
            ("PARTIAL", "PARTIALLY"),
            ("FULL", "FULLY"),
        ],
        default="NO_SUPPORT",
    )


def parse_useful(text: str) -> int:
    """An integer 1..5. Prefers a 'Rating: N' marker, else the last 1-5 digit."""
    m = re.search(r"(?:rating|score)\s*[:=]?\s*([1-5])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        digits = re.findall(r"[1-5]", line)
        if digits:
            return int(digits[-1])
    return 1


def segment_score(support: str, useful: int, w_support: float = 0.7, w_useful: float = 0.3) -> float:
    """Critic-weighted score for one candidate answer, in [0, 1].

    Grounding dominates (``w_support``) — Self-RAG's whole point is preferring an
    answer the evidence actually supports — with usefulness (``IsUse``, normalized
    from 1-5) as the tie-breaker.
    """
    sup = _SUPPORT_WEIGHT.get(support, 0.0)
    use = max(1, min(5, useful)) / 5.0
    return round(w_support * sup + w_useful * use, 4)


# --- critic calls (network via injected chat_fn) --------------------------

def need_retrieval(question: str, chat_fn=chat, model: str | None = None) -> bool:
    out = chat_fn(RETRIEVE_DECISION_PROMPT.format(question=question), model=model)
    return parse_retrieve(out)


def is_relevant(question: str, passage: str, chat_fn=chat, model: str | None = None) -> bool:
    out = chat_fn(RELEVANCE_PROMPT.format(question=question, passage=passage), model=model)
    return parse_relevance(out)


def support_label(question: str, answer: str, passage: str,
                  chat_fn=chat, model: str | None = None) -> str:
    out = chat_fn(
        SUPPORT_PROMPT.format(question=question, answer=answer, passage=passage), model=model
    )
    return parse_support(out)


def usefulness(question: str, answer: str, chat_fn=chat, model: str | None = None) -> int:
    out = chat_fn(USEFUL_PROMPT.format(question=question, answer=answer), model=model)
    return parse_useful(out)


__all__ = [
    "SUPPORT_LEVELS",
    "parse_retrieve",
    "parse_relevance",
    "parse_support",
    "parse_useful",
    "segment_score",
    "need_retrieval",
    "is_relevant",
    "support_label",
    "usefulness",
]
