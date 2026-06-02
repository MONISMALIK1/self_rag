"""Healthcare proof-of-concept for Self-RAG.

Why healthcare? It is the strongest argument for *Self*-RAG over plain RAG: a
clinical assistant must ground every claim in an authoritative, citable source and
must **refuse to answer** when the corpus does not actually cover the question. A
confident wrong dose is far worse than an honest "I don't know."

This script runs the real ``self_rag`` pipeline over a small corpus of citable
clinical reference facts (``health_corpus.jsonl``). It shows three behaviors:

  1. ANSWER  - a question the corpus covers -> grounded, cited, one-sentence answer.
  2. FILTER  - off-domain distractor passages are dropped by the relevance critic.
  3. ABSTAIN - a question the corpus does NOT cover -> the system declines.

Two execution modes, chosen automatically:

  * LIVE   - if a backend is configured (OPENROUTER_API_KEY, or SELFRAG_BASE_URL for
             a local model), the real LLM makes the reflection judgements.
  * OFFLINE- otherwise, a deterministic *extractive* stand-in critic is used. It does
             not invent medicine: generated answers are drawn verbatim from the cited
             passage, and support is judged by checking the answer's words against
             that passage. This lets the full control flow (retrieve -> relevance ->
             generate -> support/useful -> select / abstain) run with zero network.

Run:  python -m self_rag.examples.healthcare.health_demo
  or: python examples/healthcare/health_demo.py   (from the repo root)

NOT MEDICAL ADVICE. Illustrative reference snippets only; provenance is named inside
each passage. A real clinical tool needs validated sources, a dense/hybrid retriever,
a trustworthy base model, PHI-safe (local/BAA) inference, and a clinician in the loop.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Allow running both as a module (-m) and as a bare script from the repo root.
# parents: [0] healthcare  [1] examples  [2] self_rag  [3] the dir that holds it.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from self_rag.core import answer
from self_rag.corpus import Document
from self_rag.retriever import BM25Retriever, tokenize

CORPUS_PATH = Path(__file__).with_name("health_corpus.jsonl")

# ----------------------------------------------------------------------------- #
# Offline extractive critic — a deterministic stand-in for the LLM.
# It reflects only what the passage actually says; it never invents facts.
# ----------------------------------------------------------------------------- #

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "for", "to", "in", "on",
    "and", "or", "what", "which", "how", "can", "could", "should", "would", "do",
    "does", "did", "with", "at", "be", "my", "you", "your", "it", "that", "this",
    "as", "by", "from", "not", "no", "any", "about", "give", "given", "take",
    "taking", "safe", "during", "patient", "patients", "recommended",
}


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 2}


def _passage_in(prompt: str) -> str:
    m = re.search(r'"""(.*?)"""', prompt, re.S)
    return m.group(1).strip() if m else ""


def _field(prompt: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.+)", prompt)
    return m.group(1).strip() if m else ""


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _step(prompt: str) -> str:
    if "requires looking up external documents" in prompt:
        return "retrieve"
    if "relevant to a question" in prompt:
        return "relevance"
    if "using ONLY the passage" in prompt:
        return "generate"
    if "concisely, in one sentence" in prompt:
        return "direct"
    if "supported by a source passage" in prompt:
        return "support"
    if "Rate how well the answer responds" in prompt:
        return "useful"
    return "?"


def extractive_critic(prompt: str, model: str | None = None) -> str:
    step = _step(prompt)

    if step == "retrieve":
        return "RETRIEVE"  # clinical questions always warrant looking up a source

    if step == "relevance":
        q = _content_terms(_field(prompt, "Question"))
        passage = _content_terms(_passage_in(prompt))
        # A distractor shares no clinical terms with the question -> dropped.
        return "RELEVANT" if (q & passage) else "IRRELEVANT"

    if step == "generate":
        q = _content_terms(_field(prompt, "Question"))
        passage = _passage_in(prompt)
        cite_m = re.search(r"Passage \[(\d+)\]", prompt)
        cite = cite_m.group(1) if cite_m else "1"
        scored = [(len(q & _content_terms(s)), s) for s in _sentences(passage)
                  if not s.lower().startswith("source")]
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] == 0:
            return "INSUFFICIENT"
        return f"{scored[0][1]} [{cite}]"

    if step == "support":
        ans = _content_terms(re.sub(r"\[\d+\]", "", _field(prompt, "Answer")))
        passage = _content_terms(_passage_in(prompt))
        if not ans:
            return "NO_SUPPORT"
        covered = len(ans & passage) / len(ans)
        return "FULLY" if covered >= 0.8 else ("PARTIALLY" if covered >= 0.4 else "NO_SUPPORT")

    if step == "useful":
        q = _content_terms(_field(prompt, "Question"))
        ans = _content_terms(re.sub(r"\[\d+\]", "", _field(prompt, "Answer")))
        return f"Rating: {min(5, 2 + len(q & ans))}"

    return "?"


# ----------------------------------------------------------------------------- #
# Wiring
# ----------------------------------------------------------------------------- #

def load_corpus(path: Path) -> BM25Retriever:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        import json
        o = json.loads(line)
        docs.append(Document(id=o["id"], title=o.get("title", ""), text=o["text"]))
    return BM25Retriever(docs)


def pick_backend():
    """Return (chat_fn, label). Use the live model only if a backend is configured."""
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("SELFRAG_BASE_URL"):
        from self_rag.llm import chat
        return chat, "LIVE (real LLM reflection)"
    return extractive_critic, "OFFLINE (deterministic extractive critic, no network)"


class _Counter:
    """Wrap a chat_fn to count reflection/generation calls (i.e. model calls)."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def __call__(self, prompt, model=None):
        self.calls += 1
        return self._fn(prompt, model=model)


def run(q, retriever, chat_fn, early_exit):
    counter = _Counter(chat_fn)
    res = answer(q, retriever, k=4, max_candidates=3,
                 early_exit=early_exit, chat_fn=counter)
    return res, counter.calls


def show(res, base_calls, fast_calls) -> None:
    print(f"\nQ: {res.query}")
    if res.did_retrieve and res.retrieved:
        kept = {h.doc.id for h in res.relevant}
        hits = ", ".join(
            f"{h.doc.id}{'*' if h.doc.id in kept else '~'}" for h in res.retrieved
        )
        print(f"   retrieved: {hits}        (* kept relevant, ~ dropped)")
    if res.abstained:
        print(f"   -> ABSTAIN: I don't know - {res.reason}")
    else:
        print(f"   -> {res.answer}")
        print(f"      ({res.reason})")
    saved = base_calls - fast_calls
    note = f"  ({saved} fewer)" if saved else "  (no saving — abstained after full scan)"
    print(f"      model calls: {fast_calls} early-exit  vs  {base_calls} exhaustive{note}")


QUESTIONS = [
    # answerable from the corpus -> grounded, cited answers
    "What is the maximum daily dose of acetaminophen for adults?",
    "Is ibuprofen safe to use during pregnancy?",
    "What is the target INR range for a patient on warfarin?",
    "Should aspirin be given to a child recovering from influenza?",
    # NOT covered by the corpus -> the system must abstain, not guess
    "What are the warning signs of appendicitis?",
    "Is gabapentin used to treat anxiety?",
]


def main() -> int:
    chat_fn, label = pick_backend()
    retriever = load_corpus(CORPUS_PATH)
    print("=" * 72)
    print(f"Self-RAG healthcare demo   |   mode: {label}")
    print(f"corpus: {CORPUS_PATH.name} ({len(retriever.documents)} citable reference passages)")
    print("NOT MEDICAL ADVICE - illustrative demonstration of grounding + abstention.")
    print("=" * 72)

    base_total = fast_total = 0
    diverged = []
    for q in QUESTIONS:
        base_res, base_calls = run(q, retriever, chat_fn, early_exit=False)
        fast_res, fast_calls = run(q, retriever, chat_fn, early_exit=True)
        base_total += base_calls
        fast_total += fast_calls
        if (base_res.answer, base_res.abstained) != (fast_res.answer, fast_res.abstained):
            diverged.append(q)
        show(fast_res, base_calls, fast_calls)

    saved = base_total - fast_total
    pct = (saved / base_total * 100) if base_total else 0.0
    print("\n" + "-" * 72)
    print("Efficiency: early-exit accepts the first FULLY-supported, cited answer,")
    print("skipping relevance + critique calls on the remaining passages.")
    print(f"  total model calls:  {fast_total} early-exit  vs  {base_total} exhaustive"
          f"   ->  {saved} fewer ({pct:.0f}% less)")
    print(f"  same answers as exhaustive mode: {'yes' if not diverged else 'NO: ' + str(diverged)}")
    print("\nThe last two have no covering passage (appendicitis; gabapentin), so")
    print("Self-RAG declines instead of fabricating a clinical answer.")
    print("Caveat: telling a near-miss apart -- e.g. 'dose of gabapentin' vs an")
    print("acetaminophen-dose passage -- needs the model's judgement; the crude")
    print("OFFLINE critic can be fooled by shared words. That is the paper's point:")
    print("reflection quality is the base model's. Run with a real backend for that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
