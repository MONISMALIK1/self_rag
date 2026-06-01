"""A from-scratch BM25 retriever — pure Python, zero dependencies, deterministic.

Self-RAG is about *what the model does with* retrieved passages, so the retriever
itself stays deliberately simple and inspectable: classic Okapi BM25 (Robertson &
Walker, 1994) over an in-memory corpus. No embeddings, no vector index, no network
— which means retrieval is reproducible and the whole thing is unit-tested offline.

BM25 scores a document ``d`` for a query ``q`` as::

    score(q, d) = Σ_t  idf(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

where ``f(t,d)`` is the term frequency, ``|d|`` the document length, ``avgdl`` the
average length, and ``idf(t)`` the inverse document frequency. ``k1`` tunes term
saturation; ``b`` tunes length normalization. The idf uses the BM25+ ``log(1+…)``
form so it is always positive — a term can never push a score below zero.

Swap this out for a dense retriever and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Document

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. The one place text becomes terms."""
    return _TOKEN.findall(text.lower())


@dataclass
class Retrieved:
    """One search hit: the document, its BM25 score, and 1-based rank."""
    doc: Document
    score: float
    rank: int


class BM25Retriever:
    """Okapi BM25 over a fixed list of :class:`~self_rag.corpus.Document`.

    The index (term frequencies, document frequencies, idf, lengths) is built once
    at construction. ``search`` is a pure function of the query thereafter.
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b

        # Index each document over "title + text" so titles carry weight.
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        df: Counter[str] = Counter()
        for doc in self.documents:
            tokens = tokenize(f"{doc.title} {doc.text}")
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            df.update(tf.keys())  # +1 per distinct term per doc

        n = len(self.documents)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        # BM25+ idf: log(1 + (N - df + 0.5)/(df + 0.5)) — strictly positive.
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _score(self, query_terms: list[str], i: int) -> float:
        tf = self._tfs[i]
        dl = self._lengths[i]
        denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            total += self._idf.get(term, 0.0) * (f * (self.k1 + 1.0)) / (f + denom_norm)
        return total

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        """Return the top-``k`` documents that match ``query``, best first.

        Only documents with a positive score are returned, so an off-topic query
        yields an empty list rather than forcing irrelevant passages downstream —
        which is exactly the signal the pipeline needs to abstain.
        """
        terms = tokenize(query)
        if not terms or not self.documents:
            return []

        scored = [(self._score(terms, i), i) for i in range(len(self.documents))]
        # Sort by score desc, then original index asc for a stable, reproducible order.
        scored.sort(key=lambda s: (-s[0], s[1]))

        hits: list[Retrieved] = []
        for score, i in scored[:k]:
            if score <= 0.0:
                break
            hits.append(Retrieved(doc=self.documents[i], score=round(score, 4), rank=len(hits) + 1))
        return hits


__all__ = ["BM25Retriever", "Retrieved", "tokenize"]
