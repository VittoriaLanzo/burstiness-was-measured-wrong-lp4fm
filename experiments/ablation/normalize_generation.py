#!/usr/bin/env python3
"""Normalize bench generations to the human baseline's plain-prose envelope.

Drops the leading markdown/title headline block (the brief's echoed headline;
the human_text carries no headline line) and strips markdown syntax while
preserving every word of article prose. Quote style is left alone: the locked
metric normalizes typographic quotes itself (R1-5)."""
import json, re, sys

MD_INLINE = re.compile(r"\*\*|__|`|\[([^\]]+)\]\([^)]+\)")
MD_MARKER = re.compile(r"(?m)^(#{1,6}\s+|\s*[-*]\s+|>\s+)")

def normalize(text: str) -> str:
    lines = (text or "").split("\n")
    # drop leading headline block: markdown headers and/or a short unterminated title line
    while lines:
        s = lines[0].strip()
        if not s:
            lines.pop(0); continue
        if re.match(r"^#{1,6}\s", s) or (len(s) < 160 and s[-1:] not in ".!?\"'”’"
                                         and len(lines) > 1 and not lines[1].strip()):
            lines.pop(0); continue
        break
    t = "\n".join(lines)
    t = MD_INLINE.sub(lambda m: m.group(1) or "", t)   # unwrap bold/code/links
    t = MD_MARKER.sub("", t)                            # strip bullet/header markers
    return t.strip()

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    with open(dst, "w", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            r["text"] = normalize(r["text"])
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"normalized {n} rows -> {dst}")
