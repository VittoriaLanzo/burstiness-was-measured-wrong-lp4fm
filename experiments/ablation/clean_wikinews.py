#!/usr/bin/env python3
"""Derive the Wikinews `human_text` body from a `malteos/wikinews` `raw_text`.

This is the corpus-construction step for the era-controlled ablation (see README):
`clean(raw) -> str` reproduces the human side of `burstiness_pairs.jsonl` from the
raw wikitext, so the human corpus is regenerable rather than opaque. It is a real
MediaWiki AST parse (mwparserfromhell) with custom node handlers, verified to match
the shipped bodies exactly (100/100 on the held-out reconstruction harness).

Requires: mwparserfromhell  (`pip install mwparserfromhell`)

Usage:
    python -m experiments.ablation.clean_wikinews < raw.txt   # clean one article on stdin
    python -m experiments.ablation.clean_wikinews --selftest   # run the built-in checks

Rules reverse-engineered from the corpus:
- keep body only: drop everything from the first section heading OR the first
  {{haveyoursay}} terminator onward (note: {{HYS}} is NOT a terminator).
- {{w|A|B}} -> B, {{w|A}} -> A  (take the LAST pipe segment; matches quirks like
  {{w|X|foreign=suppress}} -> "foreign=suppress" and {{w|X|anchor=J|J}} -> "J").
- every other template -> removed.
- [[File:/Image:/Category:/CAT:/Media: ...]] -> removed; [[A|B]] -> B, [[A]] -> A (strip w:).
- <ref>...</ref> -> removed (with contents); '''/'' -> unwrapped (no space);
  ---- kept literally; every other HTML tag -> replaced by a space (so 8<sup>th</sup> -> "8 th").
- {|...|} tables are KEPT (mangled): raw markup passes through, pipes are stripped.
- <!--comments--> removed; html.unescape; drop leading *#:; list markers.
- collapse "<ws>" (adjacent double-quotes) to nothing; strip whitespace before . , ; : !
- mojibake (e.g. Su<0xEF><0xBF><0xBD>rez) is preserved, not re-decoded.
"""
import html
import re
import mwparserfromhell
from mwparserfromhell.nodes import (
    Template, Wikilink, Tag, Heading, Comment, HTMLEntity, Text, ExternalLink, Argument,
)

_DROP_NS = ("file:", "image:", "category:", "cat:", "media:")


def _txt(wikicode):
    if wikicode is None:
        return ""
    return "".join(_render(n) for n in wikicode.nodes)


def _render(node):
    if isinstance(node, Text):
        return node.value
    if isinstance(node, (Heading, Comment, Argument)):
        return ""
    if isinstance(node, HTMLEntity):
        return str(node)  # keep raw entity; html.unescape runs at the end

    if isinstance(node, Template):
        name = str(node.name).strip().lower()
        if name == "w":
            if node.params:
                return _txt(mwparserfromhell.parse(str(node.params[-1])))
            return ""
        return ""  # all other templates removed

    if isinstance(node, Wikilink):
        title = str(node.title).strip()
        low = title.lower()
        if any(low.startswith(ns) for ns in _DROP_NS):
            return ""
        if node.text is not None:
            return _txt(node.text)
        return re.sub(r'^\s*w:', '', title, flags=re.I)

    if isinstance(node, ExternalLink):
        return _txt(node.title) if node.title is not None else ""

    if isinstance(node, Tag):
        tag = str(node.tag).strip().lower()
        if tag == "ref":
            return ""
        if tag == "table" and "{|" in str(node):
            # mwparserfromhell structures {|...|} wiki tables and drops the |!- markup;
            # the original kept the raw markup, so re-parse with EVERY {| opener broken
            # (rows become plain text, no table can re-form -> always terminates) and let
            # inner tags/templates/links resolve normally. HTML <table> (no {|) falls through.
            return _txt(mwparserfromhell.parse(str(node).replace("{|", "{\x01|")))
        wm = node.wiki_markup
        if wm and set(str(wm)) <= {"'"}:          # ''' / '' formatting -> unwrap, no space
            return _txt(node.contents)
        if wm and str(wm).startswith("-"):         # ---- horizontal rule -> keep literally
            return str(node)
        return " " + _txt(node.contents) + " "     # <sup>,<br>,<b>,... -> tag becomes a space

    return ""


def clean(raw: str) -> str:
    code = mwparserfromhell.parse(raw)

    # body ends at the first section heading OR the first {{haveyoursay}} terminator
    # (mwparserfromhell leaves {|...|} tables as literal text; they are kept, not dropped)
    nodes = code.nodes
    cut = len(nodes)
    for i, n in enumerate(nodes):
        if isinstance(n, Heading):
            cut = i
            break
        if isinstance(n, Template) and str(n.name).strip().lower() == "haveyoursay":
            cut = i
            break
    body = mwparserfromhell.wikicode.Wikicode(nodes[:cut])

    text = _txt(body)

    # drop leading list/indent markers per line
    text = "\n".join(re.sub(r'^[ \t]*[*#:;]+[ \t]*', '', ln) for ln in text.split("\n"))

    # remove templates mwparserfromhell failed to parse (e.g. {{DISPLAYTITLE:<span>...}})
    while True:
        new = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if new == text:
            break
        text = new

    text = html.unescape(text)
    text = text.replace("\x01", "")            # remove table-opener sentinel
    text = text.replace("|", "")               # strip leftover table pipes ({| |- || |} -> { -  })
    text = re.sub(r'"\s*"', '', text)          # collapse adjacent double-quotes (split quotations)
    text = re.sub(r'\s+([.,;:!])', r'\1', text)  # no whitespace before . , ; : !
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _selftest():
    def _n(s):  # match the reconstruction harness normalization
        return re.sub(r'\s+', ' ', s).strip()
    cases = [
        ("{{w|Burj Khalifa}} tower", "Burj Khalifa tower"),          # {{w}} single -> page
        ("A {{w|Page|Label}} b", "A Label b"),                       # {{w}} -> last segment
        ("x {{w|P|foreign=suppress}} y", "x foreign=suppress y"),    # {{w}} last-segment quirk
        ("k [[Dubai|Dubai city]].\n== Sources ==\ngone", "k Dubai city."),  # section cut + link
        ("stay {{haveyoursay}}\nkeep? no", "stay"),                  # haveyoursay terminates
        ("''Bambi'' and <ref>n</ref>done", "Bambi and done"),       # italics unwrap, ref dropped
        ("8<sup>th</sup> min", "8 th min"),                          # html tag -> space
        ('a."\n\n"b', "a.b"),                                        # split-quote collapse
        ("a {{x}},{{y}}, and z .", "a,, and z."),                   # empty templates + space-before-punct
        ("<table><tr><td>x</td></tr></table> y", "x y"),            # HTML <table> must not recurse forever
    ]
    for raw, want in cases:
        got = _n(clean(raw))
        assert got == want, f"\n  raw : {raw!r}\n  got : {got!r}\n  want: {want!r}"
    print(f"selftest OK ({len(cases)} cases)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv[1:]:
        _selftest()
    else:
        print(clean(sys.stdin.read()))
