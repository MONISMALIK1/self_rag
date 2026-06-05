# self_rag

[![tests](https://github.com/MONISMALIK1/self_rag/actions/workflows/test.yml/badge.svg)](https://github.com/MONISMALIK1/self_rag/actions/workflows/test.yml)

A from-scratch, dependency-free implementation of **Self-RAG** — retrieval-augmented
generation that *reflects on itself*: it decides whether to retrieve, judges whether
each passage is relevant, checks whether its own answer is supported by the evidence,
and **abstains when it isn't**. The result is a RAG pipeline that knows when it
doesn't know.

> Asai, Wu, Wang, Sil, Hajishirzi (2023), *Self-RAG: Learning to Retrieve, Generate,
> and Critique through Self-Reflection.* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

## Faithful technique, pragmatic adaptation

The paper **fine-tunes** a language model to emit special *reflection tokens* —
`Retrieve`, `IsRel`, `IsSup`, `IsUse` — inline as it generates. This project does
**not** train a model. Instead it elicits those same four judgements from an
off-the-shelf model by **prompting**, and parses a single label out of each
response. Same control signals and same control flow; obtained at inference time
instead of through training.

What that buys you: it runs against any OpenAI-compatible endpoint (OpenRouter or a
local model) with zero training and zero dependencies. What it costs: several model
calls per question, and reflection quality is bounded by the base model's judgement
rather than learned reflection tokens. That trade-off is the point — it's the
technique, made runnable anywhere.

## The pipeline

```
Retrieve?  ──no──▶  answer directly (no sources)
   │ yes
BM25 search (top-k)
   │
IsRel  ── keep only passages judged relevant ──── none ──▶  ABSTAIN
   │
for each relevant passage: generate a 1-sentence, cited answer
   │
IsSup + IsUse  ── critique & score each candidate
   │
pick the highest-scoring candidate
   │
best is NO_SUPPORT? ── yes ──▶  ABSTAIN
   │ no
return the grounded, cited answer
```

Candidates are scored `0.7 · support + 0.3 · usefulness`, so a well-grounded answer
beats a merely fluent one — Self-RAG's core preference for *supported* generations.

## Install

No third-party dependencies. Python 3.11+.

```bash
git clone https://github.com/MONISMALIK1/self_rag.git
cd self_rag && pip install -e .      # optional; or just run from the parent dir
```

Point it at any OpenAI-compatible backend:

```bash
# OpenRouter (default)
export OPENROUTER_API_KEY=sk-or-...

# …or a local model — no key, no cloud
export SELFRAG_BASE_URL=http://localhost:11434/v1/chat/completions   # Ollama
export SELFRAG_MODEL=qwen2.5:7b
```

## Use

```bash
# answer from the bundled corpus
python -m self_rag "Which planet is the largest in the Solar System?"

# show the full reflection trace (retrieve decision, hits, relevance, critiques)
python -m self_rag "How many moons does Mars have?" --show-trace

# a question the corpus can't answer -> it abstains instead of guessing
python -m self_rag "What is the deepest ocean trench on Earth?"

# bring your own corpus: one {"id","title","text"} JSON object per line
python -m self_rag "..." --corpus mydocs.jsonl

# evaluate answer accuracy AND abstention on the bundled eval set
python -m self_rag --bench
```

Example trace:

```
--- trace ---
Retrieve? yes
Retrieved:
  [REL ] mars         bm25=3.91   Mars
  [drop] earth        bm25=0.74   Earth
Candidates:
  *[1] mars         support=FULLY      use=5 score=1.0    Mars has two moons, Phobos and Deimos. [1]
-------------
============================================================
Mars has two moons, Phobos and Deimos. [1]
(grounded in [1] (mars), support=FULLY)
```

## Example: a healthcare corpus

`examples/healthcare/` is a worked example of *why* abstention matters. It runs the
pipeline over a small set of citable clinical reference passages (drug doses,
contraindications, INR targets — each naming its source) and shows the three
behaviors a clinical assistant needs: **grounded, cited answers** for covered
questions; **relevance filtering** of off-topic passages; and **abstention** when
the corpus doesn't cover the question (e.g. appendicitis) — declining instead of
guessing a dose.

```bash
make demo        # or: python -m self_rag.examples.healthcare.health_demo
```

It runs **live** when a backend is configured, and otherwise falls back to a
deterministic offline critic so the full control flow is demonstrable with no
network. *Illustrative only — not medical advice.* A real clinical tool would need
validated sources, a dense/hybrid retriever, a trustworthy base model, PHI-safe
inference, and a clinician in the loop.

## Design

Everything that *decides* is pure stdlib and unit-tested offline; only the critic
and generation calls touch the network.

| Module | Responsibility |
| --- | --- |
| `retriever.py` | from-scratch **BM25** over an in-memory corpus (pure, deterministic) |
| `prompts.py` | the four reflection prompts + grounded generation |
| `critics.py` | **pure label parsers** (`IsRel`/`IsSup`/`IsUse`/`Retrieve`) + critic-weighted score |
| `core.py` | the control flow → `RAGResult`, including abstention |
| `corpus.py` | a small bundled corpus + an eval set (answerable, distractor, and abstain cases) |
| `llm.py` | backend-agnostic OpenAI-compatible client (OpenRouter or local) |

The parsers **fail safe**: an unreadable relevance verdict is treated as
*irrelevant* and unreadable support as *no support*, so ambiguity pushes the system
toward abstaining rather than bluffing.

## Test

```bash
make test        # or: python -m unittest discover -s self_rag/tests -t . -v
```

50 offline tests, no API key required — covering BM25 ranking, every reflection
parser, the scoring math, and the full control flow (answer, cite, select, and all
four abstention paths) driven by a scripted fake LLM.

## Limitations

Worth being upfront about:

- **Reflection quality is the base model's, not learned.** Because the four
  judgements come from prompting rather than trained reflection tokens, they're only
  as good as the model's zero-shot critiquing — a weak model can wave through an
  irrelevant passage or vouch for an unsupported claim.
- **Cost scales with passages.** Each relevant passage spends a generate + IsSup +
  IsUse call, so a question with several relevant hits costs several model calls.
  `--max-candidates` caps that, and `--fast` (early-exit) accepts the first
  fully-supported answer — skipping the critiques on the remaining passages, which
  on the bundled healthcare example cuts model calls by ~38% with identical answers.
- **BM25 is lexical.** Retrieval matches words, not meaning, so it misses pure
  paraphrases. Swapping in a dense retriever is a drop-in change — the reflection
  pipeline is unaffected.
- **Abstention is conservative by design.** The parsers fail safe, so a flaky model
  will abstain more often than strictly necessary. For this tool, a false "I don't
  know" is preferable to a confident wrong answer. When retrieval is simply *wrong*,
  abstaining is all this pipeline can do — **[corrective_rag](https://github.com/MONISMALIK1/corrective_rag)**
  picks up there, grading retrieval and *correcting* it (refine / fall back to an
  external source / combine) instead of giving up.

## Related

**[corrective_rag](https://github.com/MONISMALIK1/corrective_rag)** — Corrective RAG
(Yan et al., 2024), the successor to Self-RAG: it grades retrieval and corrects it
rather than only abstaining. Built on this project's BM25 retriever and LLM client.

**[hyde](https://github.com/MONISMALIK1/hyde)** — HyDE (Gao et al., 2022): improves
the *retrieval query itself* by searching with an LLM-written hypothetical answer
document. Complementary — it fixes retrieval *before* Self-RAG's reflection runs.

## License

MIT
