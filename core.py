"""The Self-RAG control flow: retrieve on demand, generate per passage, critique,
then select the best-grounded answer — or abstain.

Reference: Asai, Wu, Wang, Sil, Hajishirzi 2023, "Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection", https://arxiv.org/abs/2310.11511

The pipeline, one ``answer()`` call::

    Retrieve?  --no-->  answer directly (no sources)
        | yes
    BM25 search top-k
        |
    IsRel: keep only passages the model judges relevant   --none--> ABSTAIN
        |
    for each relevant passage: generate a 1-sentence, cited answer
        |
    IsSup + IsUse: critique each candidate, score it
        |
    pick the highest-scoring candidate
        |
    best is NO_SUPPORT?  --yes-->  ABSTAIN
        | no
    return the grounded, cited answer

Abstention is a first-class outcome: when nothing relevant is retrieved, or the
best candidate is not actually supported by its passage, the system says so
instead of guessing. Only the critic/generation calls touch the network (via the
injectable ``chat_fn``); retrieval and selection are pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .critics import is_relevant, need_retrieval, segment_score, support_label, usefulness
from .llm import chat
from .prompts import DIRECT_PROMPT, GENERATE_PROMPT
from .retriever import BM25Retriever, Retrieved


@dataclass
class Candidate:
    """One generated answer, tied to its source passage and critic scores."""
    text: str
    doc_id: str
    cite: int                 # citation number shown to the user
    support: str = "NO_SUPPORT"   # IsSup
    useful: int = 1               # IsUse, 1..5
    score: float = 0.0            # critic-weighted segment score


@dataclass
class RAGResult:
    query: str
    answer: str | None                 # None when abstaining
    abstained: bool
    reason: str                        # why this outcome (esp. for abstain)
    did_retrieve: bool = True
    retrieved: list[Retrieved] = field(default_factory=list)
    relevant: list[Retrieved] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    chosen: Candidate | None = None


_INSUFFICIENT = "INSUFFICIENT"


def _looks_insufficient(text: str) -> bool:
    return text.strip().upper().startswith(_INSUFFICIENT)


def _generate(question: str, passage: str, cite: int, chat_fn, model: str | None) -> str:
    prompt = GENERATE_PROMPT.format(question=question, passage=passage, cite=cite)
    return chat_fn(prompt, model=model).strip()


def _critique(query, hit, cite, chat_fn, model) -> Candidate | None:
    """Generate one cited answer from ``hit`` and score it, or None if the
    passage doesn't actually contain the answer."""
    text = _generate(query, hit.doc.text, cite, chat_fn, model)
    if _looks_insufficient(text) or not text:
        return None
    sup = support_label(query, text, hit.doc.text, chat_fn=chat_fn, model=model)
    use = usefulness(query, text, chat_fn=chat_fn, model=model)
    return Candidate(text=text, doc_id=hit.doc.id, cite=cite,
                     support=sup, useful=use, score=segment_score(sup, use))


def _finish(query, candidates, retrieved, relevant) -> RAGResult:
    """Pick the best-scoring candidate, or abstain if it isn't grounded."""
    if not candidates:
        return RAGResult(query=query, answer=None, abstained=True,
                         reason="no passage actually contained the answer",
                         retrieved=retrieved, relevant=relevant)
    chosen = max(candidates, key=lambda c: c.score)
    if chosen.support == "NO_SUPPORT":
        return RAGResult(query=query, answer=None, abstained=True,
                         reason="best answer was not supported by the evidence",
                         retrieved=retrieved, relevant=relevant,
                         candidates=candidates, chosen=chosen)
    return RAGResult(query=query, answer=chosen.text, abstained=False,
                     reason=f"grounded in [{chosen.cite}] ({chosen.doc_id}), support={chosen.support}",
                     retrieved=retrieved, relevant=relevant,
                     candidates=candidates, chosen=chosen)


def _answer_early_exit(query, retrieved, max_candidates, chat_fn, model) -> RAGResult:
    """Lazy variant: judge relevance and critique one passage at a time, and stop
    at the first FULLY-supported answer — saving relevance and critique calls on
    every remaining passage."""
    relevant: list[Retrieved] = []
    candidates: list[Candidate] = []
    for hit in retrieved:
        if len(relevant) >= max_candidates:
            break
        if not is_relevant(query, hit.doc.text, chat_fn=chat_fn, model=model):
            continue
        relevant.append(hit)
        cand = _critique(query, hit, len(relevant), chat_fn, model)
        if cand is None:
            continue
        candidates.append(cand)
        if cand.support == "FULLY":  # top grounding — no later answer can beat it
            break
    if not relevant:
        return RAGResult(query=query, answer=None, abstained=True,
                         reason="retrieved passages were judged irrelevant",
                         retrieved=retrieved, relevant=[])
    return _finish(query, candidates, retrieved, relevant)


def answer(
    query: str,
    retriever: BM25Retriever,
    k: int = 4,
    max_candidates: int = 3,
    model: str | None = None,
    early_exit: bool = False,
    chat_fn=chat,
) -> RAGResult:
    """Run the full Self-RAG pipeline for ``query`` against ``retriever``.

    ``k`` is the retrieval depth; ``max_candidates`` caps how many relevant
    passages get expanded into (critiqued) answers, bounding cost.

    ``early_exit`` trades a little answer-selection breadth for far fewer model
    calls: passages are judged relevant *lazily* and the first answer that is
    FULLY supported by its passage is accepted immediately — skipping relevance
    checks and critiques on the remaining passages. Since FULLY is the top
    support level, no later candidate could beat it on grounding; it might only
    be more *useful*. For latency- or cost-sensitive uses (e.g. a clinical
    lookup) that trade-off is usually worth it. Default off, preserving the
    exhaustive best-of-N selection.
    """
    # 1. Retrieve? — skip retrieval for trivial, self-contained questions.
    if not need_retrieval(query, chat_fn=chat_fn, model=model):
        text = chat_fn(DIRECT_PROMPT.format(question=query), model=model).strip()
        return RAGResult(query=query, answer=text, abstained=False,
                         reason="answered directly; no retrieval needed",
                         did_retrieve=False)

    # 2. Retrieve.
    retrieved = retriever.search(query, k=k)
    if not retrieved:
        return RAGResult(query=query, answer=None, abstained=True,
                         reason="no documents matched the query", retrieved=[])

    # Fast path: stop at the first FULLY-supported answer (fewer model calls).
    if early_exit:
        return _answer_early_exit(query, retrieved, max_candidates, chat_fn, model)

    # 3. IsRel — keep only passages judged relevant.
    relevant: list[Retrieved] = [
        hit for hit in retrieved
        if is_relevant(query, hit.doc.text, chat_fn=chat_fn, model=model)
    ]
    if not relevant:
        return RAGResult(query=query, answer=None, abstained=True,
                         reason="retrieved passages were judged irrelevant",
                         retrieved=retrieved, relevant=[])

    # 4. Generate one cited answer per relevant passage (capped) and critique it.
    candidates = [
        c for cite, hit in enumerate(relevant[:max_candidates], start=1)
        if (c := _critique(query, hit, cite, chat_fn, model)) is not None
    ]

    # 5-6. Pick the best-scoring candidate, or abstain if it isn't grounded.
    return _finish(query, candidates, retrieved, relevant)


def default_retriever() -> BM25Retriever:
    """A BM25 retriever over the bundled corpus — the CLI's default."""
    from .corpus import CORPUS
    return BM25Retriever(CORPUS)


__all__ = ["Candidate", "RAGResult", "answer", "default_retriever"]
