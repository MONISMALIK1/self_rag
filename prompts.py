"""Prompts for the four Self-RAG reflection steps, plus generation.

The paper trains a model to emit special *reflection tokens* — ``Retrieve``,
``IsRel``, ``IsSup``, ``IsUse`` — inline. We don't have that fine-tuned model, so
we elicit the same four judgements with focused prompts and parse a single label
out of each. Same control signals, obtained by prompting instead of by special
vocabulary. Each prompt asks for the label on the **last line** so parsing is robust
even when a model adds a sentence of explanation first.
"""

# Retrieve? — is external evidence needed at all.
RETRIEVE_DECISION_PROMPT = """You decide whether answering a question requires looking up external documents.

Answer NO_RETRIEVE only when the question is trivial general knowledge or simple \
arithmetic that needs no source (e.g. "What is 2 + 2?"). For anything factual, \
specific, or that you are not certain about, answer RETRIEVE.

Question: {question}

Respond with exactly one word on the last line: RETRIEVE or NO_RETRIEVE."""

# IsRel — does this passage help answer the question.
RELEVANCE_PROMPT = """You judge whether a document passage is relevant to a question — \
that is, whether it contains information useful for answering it.

Question: {question}

Passage:
\"\"\"{passage}\"\"\"

Respond with exactly one word on the last line: RELEVANT or IRRELEVANT."""

# Grounded generation from a single passage, with a citation marker.
GENERATE_PROMPT = """Answer the question using ONLY the passage below. Keep it to one sentence.
If the passage does not actually contain the answer, reply with exactly: INSUFFICIENT

Question: {question}

Passage [{cite}]:
\"\"\"{passage}\"\"\"

End your answer by citing the passage number in square brackets, e.g. [{cite}].
Answer:"""

# Direct answer when retrieval was deemed unnecessary.
DIRECT_PROMPT = """Answer the question concisely, in one sentence.

Question: {question}

Answer:"""

# IsSup — is the answer grounded in the passage.
SUPPORT_PROMPT = """You check whether an answer is supported by a source passage — whether \
the passage actually states the information the answer relies on.

Question: {question}
Answer: {answer}

Passage:
\"\"\"{passage}\"\"\"

Respond with exactly one label on the last line:
FULLY        - the passage clearly states it
PARTIALLY    - the passage hints at it but is incomplete
NO_SUPPORT   - the passage does not support it"""

# IsUse — how useful the answer is for the question.
USEFUL_PROMPT = """Rate how well the answer responds to the question, ignoring writing style.

Question: {question}
Answer: {answer}

Respond on the last line with: Rating: N
where N is an integer from 1 (useless) to 5 (fully and correctly answers it)."""

__all__ = [
    "RETRIEVE_DECISION_PROMPT",
    "RELEVANCE_PROMPT",
    "GENERATE_PROMPT",
    "DIRECT_PROMPT",
    "SUPPORT_PROMPT",
    "USEFUL_PROMPT",
]
