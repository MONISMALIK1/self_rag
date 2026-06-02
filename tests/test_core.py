"""The Self-RAG control flow, end to end, with a real BM25 retriever and a scripted
fake LLM — so it's fully offline yet exercises every branch.

Proves the behaviors the paper describes actually happen: retrieval is skipped for
self-contained questions, irrelevant passages are dropped, answers are grounded and
cited, the best-supported candidate is selected, and the system *abstains* rather
than guess when nothing relevant is retrieved or the answer isn't supported.
"""

import unittest

from self_rag.core import answer
from self_rag.corpus import Document
from self_rag.retriever import BM25Retriever


def step_of(prompt: str) -> str:
    """Identify which reflection/generation step a prompt belongs to."""
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


DOCS = [
    Document("jupiter", "Jupiter", "Jupiter is the largest planet in the Solar System."),
    Document("mars", "Mars", "Mars has two moons named Phobos and Deimos."),
    Document("earth", "Earth", "Earth has one moon called the Moon."),
    Document("python", "Python", "Python was created by Guido van Rossum in 1991."),
]


class HappyPathTests(unittest.TestCase):
    def test_grounded_cited_answer(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT" if "largest planet" in prompt else "IRRELEVANT"
            if s == "generate":
                return "Jupiter is the largest planet. [1]"
            if s == "support":
                return "FULLY"
            if s == "useful":
                return "Rating: 5"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake)
        self.assertFalse(res.abstained)
        self.assertIn("Jupiter", res.answer)
        self.assertTrue(res.did_retrieve)
        self.assertEqual(res.chosen.doc_id, "jupiter")
        self.assertEqual(res.chosen.support, "FULLY")

    def test_selects_more_useful_of_two_grounded_candidates(self):
        # Both passages share the query terms (so both are retrieved and judged
        # relevant) and each answer is supported by its own passage; usefulness
        # (IsUse) must break the tie in favor of the one that addresses the question.
        moon_docs = [
            Document("mars", "Mars", "Mars has two moons orbiting it in space."),
            Document("jupiter", "Jupiter", "Jupiter has dozens of moons orbiting it in space."),
        ]
        retr = BM25Retriever(moon_docs)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT"
            if s == "generate":
                return ("Mars has two moons. [1]" if "Mars has two moons" in prompt
                        else "Jupiter has dozens of moons. [2]")
            if s == "support":
                return "FULLY"
            if s == "useful":
                return "Rating: 5" if "two moons" in prompt else "Rating: 2"
            return "?"

        res = answer("How many moons orbiting it?", retr, chat_fn=fake)
        self.assertFalse(res.abstained)
        self.assertEqual(res.chosen.doc_id, "mars")
        self.assertIn("two", res.answer)
        self.assertGreaterEqual(len(res.candidates), 2)


class NoRetrieveTests(unittest.TestCase):
    def test_direct_answer_skips_retrieval(self):
        retr = BM25Retriever(DOCS)
        calls = []

        def fake(prompt, model=None):
            calls.append(step_of(prompt))
            if step_of(prompt) == "retrieve":
                return "NO_RETRIEVE"
            if step_of(prompt) == "direct":
                return "4"
            return "?"

        res = answer("What is 2 + 2?", retr, chat_fn=fake)
        self.assertFalse(res.abstained)
        self.assertFalse(res.did_retrieve)
        self.assertEqual(res.answer, "4")
        self.assertNotIn("relevance", calls)  # never touched retrieval


class AbstentionTests(unittest.TestCase):
    def test_abstains_when_no_documents_match(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            return "RETRIEVE" if step_of(prompt) == "retrieve" else "?"

        res = answer("Tell me about underwater basket weaving", retr, chat_fn=fake)
        self.assertTrue(res.abstained)
        self.assertIn("no documents", res.reason)

    def test_abstains_when_all_irrelevant(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "IRRELEVANT"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake)
        self.assertTrue(res.abstained)
        self.assertIn("irrelevant", res.reason)
        self.assertEqual(res.answer, None)

    def test_abstains_when_answer_unsupported(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT" if "largest planet" in prompt else "IRRELEVANT"
            if s == "generate":
                return "Jupiter is the largest planet. [1]"
            if s == "support":
                return "NO_SUPPORT"
            if s == "useful":
                return "Rating: 5"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake)
        self.assertTrue(res.abstained)
        self.assertIn("not supported", res.reason)
        self.assertIsNotNone(res.chosen)  # we still record what we considered

    def test_abstains_when_passage_insufficient(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT" if "largest planet" in prompt else "IRRELEVANT"
            if s == "generate":
                return "INSUFFICIENT"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake)
        self.assertTrue(res.abstained)
        self.assertIn("contained the answer", res.reason)
        self.assertEqual(res.candidates, [])


class EarlyExitTests(unittest.TestCase):
    """`early_exit=True` accepts the first FULLY-supported answer, making fewer
    model calls, while preserving the abstention guarantees."""

    @staticmethod
    def _counting(fake):
        calls = {"n": 0}

        def wrapped(prompt, model=None):
            calls["n"] += 1
            return fake(prompt, model=model)

        return wrapped, calls

    def test_early_exit_makes_fewer_calls_and_still_answers(self):
        docs = [
            Document("inr1", "Warfarin", "Warfarin target INR range is 2.0 to 3.0 for patients."),
            Document("inr2", "Anticoagulation", "The warfarin INR target range is 2.0 to 3.0 usually."),
        ]
        retr = BM25Retriever(docs)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT"
            if s == "generate":
                return "Warfarin INR target is 2.0 to 3.0. [1]"
            if s == "support":
                return "FULLY"
            if s == "useful":
                return "Rating: 5"
            return "?"

        q = "What is the warfarin INR target range?"
        w_full, c_full = self._counting(fake)
        full = answer(q, retr, chat_fn=w_full, early_exit=False)
        w_fast, c_fast = self._counting(fake)
        fast = answer(q, retr, chat_fn=w_fast, early_exit=True)

        self.assertFalse(full.abstained)
        self.assertFalse(fast.abstained)
        self.assertEqual(fast.chosen.support, "FULLY")
        self.assertLess(c_fast["n"], c_full["n"])  # stopped early -> fewer calls
        self.assertEqual(len(fast.candidates), 1)   # only the first was critiqued

    def test_early_exit_still_abstains_when_unsupported(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "RELEVANT" if "largest planet" in prompt else "IRRELEVANT"
            if s == "generate":
                return "Jupiter is the largest planet. [1]"
            if s == "support":
                return "NO_SUPPORT"
            if s == "useful":
                return "Rating: 5"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake, early_exit=True)
        self.assertTrue(res.abstained)
        self.assertIn("not supported", res.reason)

    def test_early_exit_still_abstains_when_all_irrelevant(self):
        retr = BM25Retriever(DOCS)

        def fake(prompt, model=None):
            s = step_of(prompt)
            if s == "retrieve":
                return "RETRIEVE"
            if s == "relevance":
                return "IRRELEVANT"
            return "?"

        res = answer("What is the largest planet?", retr, chat_fn=fake, early_exit=True)
        self.assertTrue(res.abstained)
        self.assertIn("irrelevant", res.reason)


if __name__ == "__main__":
    unittest.main()
