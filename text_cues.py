"""Shared transcript cues for summaries and Content 'why this score' evidence."""
from __future__ import annotations

import re
from typing import List, Sequence

# Function words + lecture small-talk that must never be treated as "topics".
_STOP = {
    "about", "after", "again", "before", "being", "could", "every", "going", "hello", "right", "there",
    "these", "those", "where", "which", "would", "your", "their", "today", "tomorrow", "really", "actually",
    "lecture", "lectures", "session", "course", "module", "topic", "students", "student", "learning",
    "usually", "people", "because", "there", "dont", "don't", "come", "comes", "coming", "have", "lots",
    "lot", "very", "just", "like", "want", "know", "think", "kind", "sort", "thing", "things", "okay",
    "yeah", "gonna", "something", "anything", "nothing", "someone", "everyone", "maybe", "probably",
    "little", "much", "more", "some", "also", "into", "from", "with", "this", "that", "them", "then",
    "than", "when", "what", "will", "were", "been", "here", "they", "we're", "we're", "thats", "that's",
    "theres", "there's", "week", "weeks", "next", "last", "today", "sorry", "alright", "anyway",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "good", "news", "nice", "great", "thanks", "thank",
    "people", "person", "class", "classes", "slide", "slides", "um", "uh",
}

_ADMIN_OPENING = re.compile(
    r"\b(don't come|do not come|attendance|midterm|midterms|week\s+\w+|people don't|running late|"
    r"can you hear|welcome back)\b",
    re.I,
)
_COURSE_CODE = re.compile(r"\b[A-Za-z]{2,}\d{2,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def salient_content_terms(text: str, limit: int = 8) -> List[str]:
    """Topical tokens: course codes first, then content nouns — never 'usually' / 'seven'."""
    raw = text or ""
    out: List[str] = []
    for code in _COURSE_CODE.findall(raw):
        key = code.upper()
        if key not in {x.upper() for x in out}:
            out.append(code.upper() if code.isupper() or re.search(r"\d", code) else code)
        if len(out) >= limit:
            return out[:limit]

    words = re.findall(r"\b[A-Za-z][a-zA-Z0-9-]{3,}\b", raw)
    for w in words:
        wl = w.lower()
        if wl in _STOP:
            continue
        if wl.endswith("ly") and wl not in {"probability", "complexity", "assembly"}:
            continue
        if not re.search(r"[aeiouy]", wl):
            continue
        if wl not in {x.lower() for x in out}:
            out.append(w if w[0].isupper() else wl)
        if len(out) >= limit:
            break
    return out[:limit]


def contentful_hook_sentence(text: str, max_chars: int = 180) -> str:
    """Prefer a sentence with a course code or disciplinary noun over attendance small-talk."""
    tex = re.sub(r"\s+", " ", (text or "").strip())
    if not tex:
        return ""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(tex) if s.strip()]
    if not parts:
        parts = [tex]

    def _clip(s: str) -> str:
        s = s.strip()
        if len(s) > max_chars:
            return s[: max_chars - 1] + "…"
        return s

    for s in parts:
        if _COURSE_CODE.search(s) and len(s.split()) >= 5:
            return _clip(s)
    for s in parts:
        if _ADMIN_OPENING.search(s) and not _COURSE_CODE.search(s):
            continue
        terms = salient_content_terms(s, 3)
        if terms and len(s.split()) >= 6:
            return _clip(s)
    for s in parts:
        if len(s.split()) >= 8:
            return _clip(s)
    return _clip(parts[0])


def contentful_evidence_sentences(text: str, n: int = 2) -> List[str]:
    """Short verbatim cues for the PDF narrative (not the first attendance joke if avoidable)."""
    tex = re.sub(r"\s+", " ", (text or "").strip())
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(tex) if s.strip()]
    picked: List[str] = []
    for s in parts:
        if len(s.split()) < 6:
            continue
        if _ADMIN_OPENING.search(s) and not _COURSE_CODE.search(s):
            continue
        if salient_content_terms(s, 1) or _COURSE_CODE.search(s):
            picked.append(s[:160])
        if len(picked) >= n:
            break
    if len(picked) < n:
        for s in parts:
            if s[:160] in picked:
                continue
            if len(s.split()) >= 6:
                picked.append(s[:160])
            if len(picked) >= n:
                break
    return picked[:n]


def snippet_around(text: str, needle: str, radius: int = 110) -> str:
    """Transcript excerpt around a phrase; requires a real word/phrase match, not a substring of another word."""
    if not text or not needle:
        return ""
    t = text
    n = needle.strip()
    if not n:
        return ""
    if re.search(r"\s", n) or len(n) >= 5:
        idx = t.lower().find(n.lower())
    else:
        m = re.search(r"\b" + re.escape(n) + r"\b", t, flags=re.I)
        idx = m.start() if m else -1
    if idx < 0:
        return ""
    lo = max(0, idx - radius)
    hi = min(len(t), idx + len(n) + radius)
    snip = t[lo:hi].replace("\n", " ").strip()
    if lo > 0:
        snip = "…" + snip
    if hi < len(t):
        snip = snip + "…"
    return snip


def first_usable_snippet(text: str, markers: Sequence[str]) -> str:
    for m in markers:
        sn = snippet_around(text, m)
        if sn and len(sn) >= 24:
            return sn
    return ""


def evidence_with_example(
    intro: str,
    transcript: str,
    markers: Sequence[str],
    empty_note: str,
) -> str:
    sn = first_usable_snippet(transcript, markers)
    if sn:
        return f'{intro} For example: "{sn}"'
    return f"{intro} {empty_note}".strip()
