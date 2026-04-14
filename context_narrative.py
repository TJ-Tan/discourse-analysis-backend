"""
Instructor-facing copy for lecture-context vs transcript alignment (no raw alignment_score jargon).
"""

from __future__ import annotations

import hashlib
import re
from typing import Set


def _theme_tags(text: str) -> Set[str]:
    t = (text or "").lower()
    tags: Set[str] = set()
    if any(x in t for x in ("bim", "building information", "revit", "ifc")):
        tags.add("BIM / building information modelling")
    if any(x in t for x in ("japanese", "japan", "nihongo", "hiragana", "katakana", "kanji", "jlpt")):
        tags.add("Japanese language")
    if any(x in t for x in ("python", "javascript", "programming", "algorithm", "software")):
        tags.add("programming / software")
    if any(x in t for x in ("structural", "civil engineering", "mechanics", "stress", "beam")):
        tags.add("structural / civil engineering")
    return tags


def _evidence_snippet(transcript_excerpt: str, max_len: int = 200) -> str:
    tex = (transcript_excerpt or "").strip().replace("\n", " ")
    if not tex:
        return ""
    first = (tex.split(".")[0] or tex).strip()
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "…"
    return first


def human_context_mismatch_paragraph(
    lecture_context: str,
    transcript_excerpt: str,
    penalty_points: float,
) -> str:
    """
    Short, conversational note for summaries and UI when a context misalignment penalty applies.
    """
    lc = (lecture_context or "").strip()
    tex = transcript_excerpt or ""
    pen = int(round(float(penalty_points or 0)))
    if pen <= 0:
        pen = 5

    ctx_tags = _theme_tags(lc)
    txt_tags = _theme_tags(tex)
    quote = _evidence_snippet(tex)

    if ctx_tags and txt_tags and ctx_tags.isdisjoint(txt_tags):
        a = sorted(ctx_tags)[0]
        b = sorted(txt_tags)[0]
        mismatch = (
            f"what you described for this session ({a}) does not match what we hear in the recording, "
            f"which centres on {b}"
        )
    elif ctx_tags:
        words = set(re.findall(r"[a-z0-9]{4,}", lc.lower()))
        junk = {
            "this", "that", "with", "from", "have", "been", "will", "your", "lecture", "session", "course",
            "students", "student", "learning", "module", "topic", "about", "into", "their", "what", "when",
            "where", "which", "there", "these", "those",
        }
        words -= junk
        tex_low = tex.lower()
        hits = sum(1 for w in list(words)[:50] if w in tex_low)
        if hits <= 1:
            mismatch = (
                "the focus you set out in the lecture context and the themes in the transcript excerpt look like different topics"
            )
        else:
            mismatch = (
                "the lecture context and the transcript still read as pointing to different primary aims for this recording"
            )
    else:
        mismatch = (
            "the lecture context you supplied and what the transcript excerpt actually covers do not line up clearly"
        )

    ev = f' Evidence includes: "{quote}".' if quote else ""

    h = int(hashlib.sha256(f"{lc}|{tex}".encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
    openings = (
        f"In our Context-Aware Analysis, we found that {mismatch}.",
        f"In our Context-Aware Analysis, the clearest signal is that {mismatch}.",
        f"Our Context-Aware Analysis highlights that {mismatch}.",
    )
    lead = openings[h % len(openings)]

    closings = (
        f"Because of this mismatch, the Content score was reduced by −{pen} point(s) after rubric scoring "
        f"(see the scoring breakdown for the figures used in the overall MARS calculation).",
        f"Given that gap, the Content score was adjusted by −{pen} point(s) after rubric scoring "
        f"(the breakdown shows how this feeds the overall MARS figure).",
        f"That misalignment triggered a −{pen} Content adjustment after rubric scoring "
        f"(see the score breakdown for how it combines with the other pillars).",
    )
    tail = closings[(h // 7) % len(closings)]

    return f"{lead}{ev} {tail}"
