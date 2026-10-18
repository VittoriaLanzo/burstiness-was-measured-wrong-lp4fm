#!/usr/bin/env python3
"""Tests for the contamination probe (offline, synthetic texts)."""
from __future__ import annotations

import unittest

from experiments.ablation.contamination_probe import ngrams, overlap, probe_model


class NgramOverlap(unittest.TestCase):
    def test_verbatim_copy_is_full_overlap(self):
        t = "the quick brown fox jumps over the lazy dog again and again today"
        self.assertEqual(overlap(t, t, 8), 1.0)

    def test_disjoint_texts_zero(self):
        a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        b = "one two three four five six seven eight nine ten eleven"
        self.assertEqual(overlap(a, b, 8), 0.0)

    def test_short_text_no_crash(self):
        self.assertEqual(overlap("too short", "reference text here", 8), 0.0)
        self.assertEqual(ngrams("a b c", 8), set())


class ProbeModel(unittest.TestCase):
    def test_memorizer_flagged_clean_model_not(self):
        human = {str(i): f"unique human article {i} " + " ".join(f"w{i}x{j}" for j in range(30))
                 for i in range(4)}
        clean = {sid: [f"fresh model text {sid} " + " ".join(f"g{sid}y{j}" for j in range(30))]
                 for sid in human}
        memorizer = {sid: [human[sid]] for sid in human}  # copies its paired article
        r_clean = probe_model(clean, human, n_sizes=(8,))
        r_mem = probe_model(memorizer, human, n_sizes=(8,))
        self.assertEqual(r_clean["8gram"]["matched_mean"], 0.0)
        self.assertEqual(r_mem["8gram"]["matched_mean"], 1.0)
        self.assertLess(r_mem["8gram"]["mismatched_mean"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
