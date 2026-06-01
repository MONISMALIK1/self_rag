"""CLI for Self-RAG.

Usage:
    # Answer one question against the bundled corpus
    python -m self_rag "Which planet is the largest in the Solar System?"

    # Show the full reflection trace: retrieve decision, hits, relevance, critiques
    python -m self_rag "How many moons does Mars have?" --show-trace

    # A question the corpus can't answer -> the system abstains instead of guessing
    python -m self_rag "What is the deepest ocean trench on Earth?"

    # Point it at your own corpus (one {"id","title","text"} JSON object per line)
    python -m self_rag "..." --corpus mydocs.jsonl

    # Benchmark answer accuracy + abstention on the bundled eval set
    python -m self_rag --bench
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import answer, default_retriever
from .corpus import EVAL_QUESTIONS, Document, matches
from .llm import DEFAULT_MODEL
from .retriever import BM25Retriever


def _load_corpus(path: str) -> BM25Retriever:
    docs: list[Document] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(Document(id=str(obj["id"]), title=obj.get("title", ""), text=obj["text"]))
    if not docs:
        raise SystemExit(f"no documents loaded from {path}")
    return BM25Retriever(docs)


def _print_trace(res) -> None:
    print("--- trace ---")
    print(f"Retrieve? {'yes' if res.did_retrieve else 'no (answered directly)'}")
    if res.retrieved:
        print("Retrieved:")
        rel_ids = {h.doc.id for h in res.relevant}
        for h in res.retrieved:
            mark = "REL " if h.doc.id in rel_ids else "drop"
            print(f"  [{mark}] {h.doc.id:<12} bm25={h.score:<7} {h.doc.title}")
    if res.candidates:
        print("Candidates:")
        for c in res.candidates:
            star = " *" if res.chosen is c else "  "
            head = c.text.replace("\n", " ")
            if len(head) > 60:
                head = head[:57] + "..."
            print(f" {star}[{c.cite}] {c.doc_id:<12} support={c.support:<10} "
                  f"use={c.useful} score={c.score:<6} {head}")
    print("-------------")


def _answer_one(args, retriever) -> int:
    res = answer(args.query, retriever, k=args.k, max_candidates=args.max_candidates,
                 model=args.model)
    if args.show_trace:
        _print_trace(res)
    print("=" * 60)
    if res.abstained:
        print(f"I don't know — {res.reason}")
    else:
        print(res.answer)
        print(f"({res.reason})")
    return 0


def _bench(args, retriever) -> int:
    answerable_total = answerable_ok = 0
    abstain_total = abstain_ok = 0

    for i, ex in enumerate(EVAL_QUESTIONS, 1):
        res = answer(ex.question, retriever, k=args.k,
                     max_candidates=args.max_candidates, model=args.model)
        should_abstain = ex.answer is None

        if should_abstain:
            abstain_total += 1
            ok = res.abstained
            abstain_ok += int(ok)
            verdict = "ABSTAIN-OK" if ok else "SHOULD-ABSTAIN"
            shown = "(abstained)" if res.abstained else (res.answer or "")
        else:
            answerable_total += 1
            ok = (not res.abstained) and matches(res.answer, ex)
            answerable_ok += int(ok)
            verdict = "OK" if ok else ("ABSTAINED" if res.abstained else "WRONG")
            shown = "(abstained)" if res.abstained else (res.answer or "")

        head = shown.replace("\n", " ")
        if len(head) > 48:
            head = head[:45] + "..."
        print(f"[{i:2d}] {verdict:<14} gold={str(ex.answer):<18} {head}", flush=True)

    total = answerable_total + abstain_total
    correct = answerable_ok + abstain_ok
    print("\n" + "=" * 64)
    print(f"Self-RAG on bundled eval set — model={args.model or DEFAULT_MODEL}")
    print("=" * 64)
    print(f"  Answerable:  {answerable_ok}/{answerable_total} correct")
    print(f"  Abstentions: {abstain_ok}/{abstain_total} correct (declined when unanswerable)")
    print(f"  Overall:     {correct}/{total} = {correct / total * 100:.1f}%")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="self_rag",
        description="Self-RAG (Asai et al., 2023): retrieve on demand, generate, "
                    "self-critique for relevance and support, and abstain when unsure.",
    )
    p.add_argument("query", nargs="?", help="The question to answer.")
    p.add_argument("--k", type=int, default=4, help="Passages to retrieve (default: 4).")
    p.add_argument("--max-candidates", type=int, default=3,
                   help="Relevant passages to expand into answers (default: 3).")
    p.add_argument("--model", default=None, help=f"Model slug (default: {DEFAULT_MODEL}).")
    p.add_argument("--corpus", default=None,
                   help="Path to a JSONL corpus ({id,title,text} per line); "
                        "defaults to the bundled corpus.")
    p.add_argument("--show-trace", action="store_true",
                   help="Print the retrieve/relevance/critique trace.")
    p.add_argument("--bench", action="store_true",
                   help="Evaluate answer accuracy + abstention on the bundled eval set.")
    args = p.parse_args()

    retriever = _load_corpus(args.corpus) if args.corpus else default_retriever()

    if args.bench:
        return _bench(args, retriever)

    if not args.query:
        p.error("provide a question to answer, or use --bench")

    print(f"\nQuestion: {args.query}", file=sys.stderr)
    print(f"Model: {args.model or DEFAULT_MODEL}\n", file=sys.stderr, flush=True)
    return _answer_one(args, retriever)


if __name__ == "__main__":
    raise SystemExit(main())
