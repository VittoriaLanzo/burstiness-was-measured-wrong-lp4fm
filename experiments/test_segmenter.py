#!/usr/bin/env python3
"""Regression tests for the sentence segmenter shared by every metric."""
from __future__ import annotations

import unittest

from experiments.burstiness_metrics import sentence_lengths


class CurlyQuoteBoundaries(unittest.TestCase):
    def test_curly_closing_quote_splits_like_straight(self):
        curly = "He said “yes.” Then he left. She asked “why?” Nobody knew."
        straight = 'He said "yes." Then he left. She asked "why?" Nobody knew.'
        self.assertEqual(sentence_lengths(curly), sentence_lengths(straight))
        self.assertEqual(sentence_lengths(curly), [3, 3, 3, 2])

    def test_curly_single_quote_and_apostrophe(self):
        t = "It’s late, he said ‘no.’ We went home."
        self.assertEqual(sentence_lengths(t), [5, 3])

    def test_plain_prose_unchanged(self):
        t = "A plain sentence here. Another one follows it. And a third."
        self.assertEqual(sentence_lengths(t), [4, 4, 3])

    def test_abbreviation_guard_still_applies(self):
        self.assertEqual(sentence_lengths("Mr. Smith arrived. He sat."), [3, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
