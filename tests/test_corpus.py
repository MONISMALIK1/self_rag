"""Bundled corpus + eval-set integrity, and the answer-matching helper."""

import unittest

from self_rag.core import default_retriever
from self_rag.corpus import CORPUS, EVAL_QUESTIONS, QAExample, matches


class CorpusTests(unittest.TestCase):
    def test_ids_unique(self):
        ids = [d.id for d in CORPUS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_documents_nonempty(self):
        for d in CORPUS:
            self.assertTrue(d.id and d.title and d.text.strip(), f"empty fields in {d!r}")

    def test_default_retriever_indexes_corpus(self):
        r = default_retriever()
        self.assertEqual(len(r.documents), len(CORPUS))
        hits = r.search("Great Red Spot storm", k=3)
        self.assertEqual(hits[0].doc.id, "jupiter")


class EvalSetTests(unittest.TestCase):
    def test_has_answerable_and_abstain(self):
        answerable = [q for q in EVAL_QUESTIONS if q.answer is not None]
        abstain = [q for q in EVAL_QUESTIONS if q.answer is None]
        self.assertGreaterEqual(len(answerable), 3)
        self.assertGreaterEqual(len(abstain), 1)


class MatchesTests(unittest.TestCase):
    def test_exact_and_substring(self):
        ex = QAExample("q", "Jupiter")
        self.assertTrue(matches("The answer is Jupiter.", ex))
        self.assertFalse(matches("The answer is Saturn.", ex))

    def test_alias(self):
        ex = QAExample("q", "two", aliases=("2",))
        self.assertTrue(matches("It has 2 moons.", ex))

    def test_case_and_punctuation_insensitive(self):
        ex = QAExample("q", "Guido van Rossum")
        self.assertTrue(matches("Created by guido van rossum!", ex))

    def test_none_predicted_or_abstain_example(self):
        self.assertFalse(matches(None, QAExample("q", "x")))
        self.assertFalse(matches("anything", QAExample("q", None)))


if __name__ == "__main__":
    unittest.main()
