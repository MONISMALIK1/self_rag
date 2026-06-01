"""BM25 retriever — pure, deterministic, offline.

Proves the retriever ranks on-topic documents first, returns nothing for an
off-topic query (the signal the pipeline uses to abstain), respects top-k, and is
reproducible.
"""

import unittest

from self_rag.corpus import Document
from self_rag.retriever import BM25Retriever, tokenize

DOCS = [
    Document("cats", "Cats", "Cats are small domesticated felines that purr and hunt mice."),
    Document("dogs", "Dogs", "Dogs are loyal domesticated canines that bark and fetch."),
    Document("cars", "Cars", "A car is a wheeled motor vehicle used for transport on roads."),
]


class TokenizeTests(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Hello, World! 42 cats."), ["hello", "world", "42", "cats"])

    def test_empty(self):
        self.assertEqual(tokenize("   !!!  "), [])


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.r = BM25Retriever(DOCS)

    def test_ranks_on_topic_first(self):
        hits = self.r.search("loyal canines that bark", k=3)
        self.assertEqual(hits[0].doc.id, "dogs")
        self.assertEqual(hits[0].rank, 1)

    def test_off_topic_returns_nothing(self):
        # No shared terms -> no positive score -> empty, so the pipeline can abstain.
        self.assertEqual(self.r.search("submarine periscope", k=4), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.r.search("", k=4), [])

    def test_only_matching_docs_returned(self):
        # "felines" appears in exactly one doc.
        hits = self.r.search("felines", k=4)
        self.assertEqual([h.doc.id for h in hits], ["cats"])

    def test_respects_k(self):
        # "domesticated" is in cats and dogs; asking for k=1 returns one.
        hits = self.r.search("domesticated", k=1)
        self.assertEqual(len(hits), 1)

    def test_deterministic(self):
        a = self.r.search("domesticated animals", k=3)
        b = self.r.search("domesticated animals", k=3)
        self.assertEqual([(h.doc.id, h.score) for h in a],
                         [(h.doc.id, h.score) for h in b])

    def test_empty_corpus(self):
        self.assertEqual(BM25Retriever([]).search("anything", k=4), [])


if __name__ == "__main__":
    unittest.main()
