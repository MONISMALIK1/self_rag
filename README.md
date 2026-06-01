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

## License

MIT
