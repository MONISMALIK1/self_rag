"""The reflection layer — pure label parsers + the critic-weighted score.

These are the deterministic heart of Self-RAG's self-reflection: turning a model's
free text into Retrieve / IsRel / IsSup / IsUse decisions. The parsers must read
the intended label out of messy output and fail *safe* when they can't.
"""

import unittest

from self_rag.critics import (
    is_relevant,
    need_retrieval,
    parse_relevance,
    parse_retrieve,
    parse_support,
    parse_useful,
    segment_score,
    support_label,
    usefulness,
)


class ParseRetrieveTests(unittest.TestCase):
    def test_plain_labels(self):
        self.assertTrue(parse_retrieve("RETRIEVE"))
        self.assertFalse(parse_retrieve("NO_RETRIEVE"))

    def test_with_explanation_last_line_wins(self):
        self.assertFalse(parse_retrieve("This is trivial.\nDecision: NO_RETRIEVE"))
        self.assertTrue(parse_retrieve("Need a source.\nRETRIEVE"))

    def test_defaults_to_retrieve_when_unsure(self):
        self.assertTrue(parse_retrieve("hmm not sure"))


class ParseRelevanceTests(unittest.TestCase):
    def test_plain_labels(self):
        self.assertTrue(parse_relevance("RELEVANT"))
        self.assertFalse(parse_relevance("IRRELEVANT"))

    def test_irrelevant_not_confused_with_relevant(self):
        self.assertFalse(parse_relevance("This passage is IRRELEVANT to the question."))
        self.assertFalse(parse_relevance("not relevant"))

    def test_defaults_to_irrelevant_when_unsure(self):
        # Fail safe: drop a passage we can't confirm is relevant.
        self.assertFalse(parse_relevance("uh, maybe?"))


class ParseSupportTests(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(parse_support("FULLY"), "FULLY")
        self.assertEqual(parse_support("PARTIALLY"), "PARTIALLY")
        self.assertEqual(parse_support("NO_SUPPORT"), "NO_SUPPORT")

    def test_phrasings(self):
        self.assertEqual(parse_support("The passage fully supports it."), "FULLY")
        self.assertEqual(parse_support("It is partially supported."), "PARTIALLY")
        self.assertEqual(parse_support("The passage does not support the answer."), "NO_SUPPORT")

    def test_defaults_to_no_support(self):
        self.assertEqual(parse_support("???"), "NO_SUPPORT")


class ParseUsefulTests(unittest.TestCase):
    def test_rating_marker(self):
        self.assertEqual(parse_useful("Rating: 4"), 4)
        self.assertEqual(parse_useful("Rating: 5 out of 5"), 5)

    def test_bare_digit(self):
        self.assertEqual(parse_useful("I'd give it a 3."), 3)

    def test_defaults_to_one(self):
        self.assertEqual(parse_useful("no number here"), 1)


class SegmentScoreTests(unittest.TestCase):
    def test_fully_and_useful_is_max(self):
        self.assertEqual(segment_score("FULLY", 5), 1.0)

    def test_no_support_caps_low(self):
        # Grounding dominates: zero support leaves only the usefulness term.
        self.assertEqual(segment_score("NO_SUPPORT", 5), 0.3)

    def test_partial_between(self):
        self.assertEqual(segment_score("PARTIALLY", 5), 0.65)

    def test_grounding_outranks_usefulness(self):
        # A fully-supported, mediocre answer beats an unsupported, "useful" one.
        self.assertGreater(segment_score("FULLY", 2), segment_score("NO_SUPPORT", 5))


class CriticCallTests(unittest.TestCase):
    """The critic functions just wire a chat_fn to a parser; inject a fake."""

    def test_need_retrieval(self):
        self.assertFalse(need_retrieval("q", chat_fn=lambda p, model=None: "NO_RETRIEVE"))
        self.assertTrue(need_retrieval("q", chat_fn=lambda p, model=None: "RETRIEVE"))

    def test_is_relevant(self):
        self.assertTrue(is_relevant("q", "p", chat_fn=lambda p, model=None: "RELEVANT"))
        self.assertFalse(is_relevant("q", "p", chat_fn=lambda p, model=None: "IRRELEVANT"))

    def test_support_label(self):
        self.assertEqual(support_label("q", "a", "p", chat_fn=lambda p, model=None: "FULLY"),
                         "FULLY")

    def test_usefulness(self):
        self.assertEqual(usefulness("q", "a", chat_fn=lambda p, model=None: "Rating: 4"), 4)


if __name__ == "__main__":
    unittest.main()
