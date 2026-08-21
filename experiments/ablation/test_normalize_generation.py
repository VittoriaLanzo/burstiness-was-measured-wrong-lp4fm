#!/usr/bin/env python3
"""Tests for the generation normalizer (markdown/title scaffolding -> plain prose)."""
from __future__ import annotations

import unittest

from experiments.ablation.normalize_generation import normalize

GEMMA_SHAPE = ("## Fire Breaks Out at Dubai Hotel\n\n**Dubai, UAE – December 31** – A "
               "significant fire erupted at the hotel tonight. Guests were evacuated "
               "as emergency services worked to contain the blaze.")


class Normalize(unittest.TestCase):
    def test_drops_md_headline_and_unwraps_bold(self):
        out = normalize(GEMMA_SHAPE)
        self.assertNotIn("##", out)
        self.assertNotIn("**", out)
        self.assertTrue(out.startswith("Dubai, UAE"))
        self.assertIn("emergency services", out)

    def test_plain_prose_untouched(self):
        t = "A plain article sentence. Another one follows it. And a third."
        self.assertEqual(normalize(t), t)

    def test_bare_title_line_dropped(self):
        t = "Council approves the new bridge\n\nThe council voted on Friday. Work begins soon."
        self.assertTrue(normalize(t).startswith("The council voted"))

    def test_bullets_unwrapped_words_kept(self):
        t = "The plan has parts.\n- first item here\n- second item here\nIt ends."
        out = normalize(t)
        self.assertNotIn("- ", out)
        self.assertIn("first item here", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
