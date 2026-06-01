"""Self-RAG: retrieval-augmented generation that critiques itself (Asai et al., 2023).

Ordinary RAG retrieves some passages and hopes the answer is in there. Self-RAG
adds *self-reflection*: the model decides whether to retrieve at all, judges
whether each passage is actually relevant, checks whether its own answer is
supported by the evidence, and — crucially — **abstains** when it isn't. The
result is a RAG pipeline that knows when it doesn't know.

The paper fine-tunes a model to emit reflection tokens (Retrieve / IsRel / IsSup /
IsUse) inline. Lacking that model, this implementation elicits the same four
judgements by prompting and parses one label out of each — the technique without
the training. See the README for that faithful-vs-pragmatic distinction.

Everything that *decides* — BM25 retrieval, the reflection-label parsers, the
critic-weighted selection — is pure stdlib and unit-tested offline; only the
critic and generation calls touch the network.

Public API:
    answer(query, retriever, ...)          # the full pipeline -> RAGResult
    RAGResult / Candidate                  # the outcome + per-candidate critiques
    BM25Retriever / Retrieved / tokenize   # pure-Python retrieval
    default_retriever()                    # BM25 over the bundled corpus
    CORPUS / EVAL_QUESTIONS / Document      # bundled data + eval set
    need_retrieval / is_relevant / support_label / usefulness   # the critics
    parse_* / segment_score                 # the pure reflection-label layer
"""

from .core import Candidate, RAGResult, answer, default_retriever
from .corpus import CORPUS, EVAL_QUESTIONS, Document, QAExample, matches
from .critics import (
    SUPPORT_LEVELS,
    is_relevant,
    need_retrieval,
    parse_relevance,
    parse_retrieve,
    parse_support,
    parse_useful,
    segment_score,
    support_label,
    usefulness,
)
from .llm import DEFAULT_MODEL, chat
from .retriever import BM25Retriever, Retrieved, tokenize

__all__ = [
    "BM25Retriever",
    "CORPUS",
    "Candidate",
    "DEFAULT_MODEL",
    "Document",
    "EVAL_QUESTIONS",
    "QAExample",
    "RAGResult",
    "Retrieved",
    "SUPPORT_LEVELS",
    "answer",
    "chat",
    "default_retriever",
    "is_relevant",
    "matches",
    "need_retrieval",
    "parse_relevance",
    "parse_retrieve",
    "parse_support",
    "parse_useful",
    "segment_score",
    "support_label",
    "tokenize",
    "usefulness",
]
