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
    lead: str,
    transcript: str,
    markers: Sequence[str],
    empty_note: str,
    quote_limit: int = 140,
) -> str:
    """Lead sentence plus a short verbatim cue, or a fallback note when no marker hits.

    Used by Content 'why this score' evidence. Must stay exported: ai_processor imports
    this name at module load, and a missing export puts the API into mock scoring.
    """
    lead_s = (lead or "").strip()
    note = (empty_note or "").strip()
    sn = first_usable_snippet(transcript or "", markers)
    if sn:
        quote = re.sub(r"\s+", " ", sn).strip()
        if len(quote) > quote_limit:
            quote = quote[: quote_limit - 1] + "…"
        if lead_s:
            return f'{lead_s} Example from the lecture: “{quote}”.'
        return f'Example from the lecture: “{quote}”.'
    if lead_s and note:
        return f"{lead_s} {note}"
    return lead_s or note


_GENERIC_ORG = {
    "school", "faculty", "university", "college", "department", "institute", "computing",
    "education", "information", "sciences", "science", "engineering", "soc",
}

# Short domain tokens and near-synonyms so "gpu" matches a GPGPU / CUDA lecture.
_CONTEXT_SYNONYMS = {
    "gpu": ("gpu", "gpgpu", "cuda", "nvidia", "graphics", "parallel", "kernel"),
    "gpgpu": ("gpgpu", "gpu", "cuda", "nvidia", "graphics"),
    "cuda": ("cuda", "gpu", "gpgpu", "nvidia"),
    "nvidia": ("nvidia", "cuda", "gpu"),
    "computational": ("computational", "compute", "computation", "computing"),
    "parallel": ("parallel", "parallelism", "gpu", "gpgpu"),
    "graphics": ("graphics", "gpu", "gpgpu"),
}


def _context_term_matches(term: str, transcript_lower: str) -> bool:
    t = (term or "").lower().strip()
    if not t:
        return False
    variants = _CONTEXT_SYNONYMS.get(t, (t,))
    tl = transcript_lower or ""
    for v in variants:
        if len(v) <= 3:
            if re.search(r"\b" + re.escape(v) + r"\b", tl) or v in tl:
                return True
        elif v in tl:
            return True
    return False


def lecture_context_alignment(lecture_context: str, transcript: str) -> dict:
    """
    Align instructor context with the transcript.

    Naive keyword overlap fails for 'computational gpu' vs a GPGPU lecture: 'gpu' is only
    three letters (dropped by 4+ filters) and 'school'/'computing' are institutional, not topical.
    """
    lc = (lecture_context or "").strip()
    t = (transcript or "").strip()
    if not lc or not t:
        return {
            "alignment_score": None,
            "verdict": None,
            "rationale": "No lecture context or transcript text available for alignment check.",
            "matched_terms": [],
            "snippet": "",
        }
    stop = {
        "this", "that", "these", "those", "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been", "being",
        "it", "we", "you", "they", "i", "our", "your", "their", "lecture", "session", "week", "module",
        "course", "topic", "learning", "outcome", "outcomes", "students", "student", "audience",
        "should", "teach", "about", "using", "use",
    }
    stop |= _GENERIC_ORG
    # Allow 3-letter technical tokens (gpu, cpu, ram, api).
    ctx_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", lc.lower())
    ctx_terms: List[str] = []
    for tok in ctx_tokens:
        if tok in stop:
            continue
        if tok not in ctx_terms:
            ctx_terms.append(tok)
        if len(ctx_terms) >= 18:
            break
    tl = t.lower()
    hits = [w for w in ctx_terms if _context_term_matches(w, tl)]
    ctx_l = lc.lower()
    ctx_has_gpu_family = any(x in ctx_l for x in ("gpu", "gpgpu", "cuda", "nvidia"))
    tr_has_gpu_family = any(_context_term_matches(h, tl) for h in ("gpu", "gpgpu", "cuda", "nvidia"))
    if ctx_has_gpu_family and tr_has_gpu_family:
        score = max(0.85, (len(hits) / max(1, len(ctx_terms))) if ctx_terms else 0.85)
        verdict = "match"
        rationale = (
            "Instructor context names GPU/GPGPU-style computing and the transcript uses the same family of terms "
            f"(matched: {', '.join(hits) or 'gpu/gpgpu/cuda'})."
        )
        snippet = snippet_around(t, hits[0] if hits else "gpu") or snippet_around(t, "gpgpu") or snippet_around(t, "cuda")
        return {
            "alignment_score": round(float(score), 3),
            "verdict": verdict,
            "rationale": rationale,
            "matched_terms": hits[:10] or ["gpu"],
            "snippet": snippet,
        }

    if not ctx_terms:
        return {
            "alignment_score": 0.5,
            "verdict": "partial",
            "rationale": "Context was mostly generic (e.g. school/course wording) so topical overlap could not be scored strictly.",
            "matched_terms": [],
            "snippet": "",
        }
    score = len(hits) / max(1, len(ctx_terms))
    if score >= 0.35:
        verdict = "match"
    elif score >= 0.15 or hits:
        # One strong topical hit (e.g. gpu) is enough for partial+, not a 'weak overlap' scare.
        verdict = "match" if hits else "partial"
        if hits and score < 0.35:
            verdict = "match"
            score = max(score, 0.4)
    else:
        verdict = "mismatch"
    snippet = ""
    if hits:
        for w in hits[:3]:
            snippet = snippet_around(t, w)
            if snippet:
                break
    rationale = (
        f"Keyword overlap between context and transcript is {len(hits)}/{len(ctx_terms)} topical terms (score={score:.2f})."
        + (f' Example matched cue: "{snippet}".' if snippet else "")
    )
    return {
        "alignment_score": round(float(score), 3),
        "verdict": verdict,
        "rationale": rationale,
        "matched_terms": hits[:10],
        "snippet": snippet,
    }


def summary_context_alignment_sentence(lecture_context: str, transcript: str, analysis_verdict: str = "") -> str:
    """Instructor-facing sentence for PDF/summary fallbacks."""
    lc = (lecture_context or "").strip()
    if not lc:
        return (
            " No instructor context was provided for this lecture (for example module, topic, or intended learning outcomes), "
            "so stated-versus-delivered alignment cannot be assessed from the submission."
        )
    v = (analysis_verdict or "").lower().strip()
    if not v:
        v = (lecture_context_alignment(lc, transcript).get("verdict") or "").lower().strip()
    if v == "match":
        return (
            " Against the instructor-supplied context, the transcript matches the stated topic "
            "(for example a GPU/GPGPU computing lecture when that is what you named)."
        )
    if v == "partial":
        return (
            " Against the instructor-supplied context, the transcript is partly aligned with the stated topic; "
            "some terms overlap and some may be implied rather than repeated verbatim."
        )
    if v == "mismatch":
        return (
            " Against the instructor-supplied context, the spoken content appears to be a different topic; "
            "check whether the recording is the session you intended to analyse."
        )
    return (
        " Instructor context was provided; topical overlap with the transcript could not be scored cleanly from keywords alone."
    )

