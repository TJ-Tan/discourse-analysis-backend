# MARS Scoring Rubric — Revised Rubric (v20260224)

Source: `MARS Rubric v20260224.xlsx`, sheet **Revised Rubric**.  
This document mirrors the spreadsheet structure, **fills empty “How MARS Works” and “Remarks” cells** with implementation notes, and aligns with the codebase (`metrics_config.MARS_CONFIG`, `ai_processor`).

---

## How MARS Works (high level)

1. **Ingest** lecture video → audio, frames, full transcript (Whisper), optional **lecture context** (subject, ILOs) from the user.
2. **Content (20%)** — Nine criteria scored 1–10 via **LLM** on transcript (+ context), grouped into Content Organisation, Explanation Quality, and Use of Examples / Representation; **causal reasoning** is blended with a **causal-marker** heuristic.
3. **Delivery (40%)** — **50% Speech** (five acoustic/ASR metrics) + **50% Body language** (five vision metrics per frame, aggregated).
4. **Engagement (40%)** — **Question density (QD)**; **CLI** (ICAP-derived); **SUI**; **QDS**; **student feedback** (learner question frequency & cognitive level) via **LLM** with low confidence → 0 when webcast has no audience audio.
5. **Overall** — `0.20×Content + 0.40×Delivery + 0.40×Engagement` (each main category is 0–10 before weighting).

---

## Main category weights (overall MARS)

| Main Category | Weight | Role in code |
|---------------|--------|----------------|
| **Content** | 0.20 | `compute_mars_content_category_score()` |
| **Delivery** | 0.40 | `compute_mars_delivery_category_score()` = 0.5×speech + 0.5×visual |
| **Engagement** | 0.40 | `compute_mars_engagement_category_score()` |

---

## Detailed rubric table

**Columns:** Main Category | Sub Category | Criteria | Definition | Reference Sources | How MARS Works (filled) | Remarks (filled)

### Content (0.20)

| Sub (weight of Content) | Criteria (weight of sub) | Definition | Reference | How MARS Works | Remarks |
|---------------------------|----------------------------|------------|-----------|----------------|---------|
| **Content Organisation** (0.30) | **Structural Sequencing** (0.10) | Logically progressive sequence from foundations to complexity. | Chi & Wylie, 2014; ICAP | LLM scores transcript; user **lecture context** sets intended sequence/ILOs. | Align prompts with ILO order when context provided. |
| | **Logical Consistency** (0.10) | Coherent reasoning without contradictions. | Chi et al., 1989 | LLM holistic pass on transcript + context. | |
| | **Closure / Framing** (0.10) | Summaries, objectives, links to framework. | Rosenshine, 2012 | LLM detects openings/closures and explicit links to learning goals. | |
| **Explanation Quality** (0.40) | **Conceptual Accuracy** (**0.20**) | Disciplinary concepts and terminology correct. | Shulman, 1986 | LLM evaluates accuracy vs domain norms; context names subject. | **0.20 of 0.40** within Explanation Quality (50% of that sub-block). |
| | **Causal Reasoning Depth** (0.10) | Cause–effect beyond description. | Chi, 2009 | LLM score **plus** blend with density of causal connectors (“because”, “therefore”, …). | Matches spreadsheet note: connector detection + LLM. |
| | **Multi-Perspective Explanation** (0.10) | Multiple models/angles on concepts. | Reiser, 2013 | LLM searches for contrasts, “another way”, multiple representations in speech. | |
| **Use of Examples / Representation** (0.30) | **Example Quality & Frequency** (0.10) | Concrete examples for abstraction. | Chi et al., 1989 | LLM rates frequency and pedagogical quality of examples. | |
| | **Analogy / Concept Bridging** (0.10) | Analogies linking to prior knowledge. | Gentner & Colhoun, 2010 | LLM + keyword cues (“like”, “similar to”, “think of”). | Spreadsheet: semantic mapping + LLM. |
| | **Representation Diversity** (0.10) | Verbal, symbolic, diagram references, etc. | Ainsworth, 2006 | LLM checks references to figures, equations, multiple modalities in speech. | Video frames inform “visual” delivery but this criterion is **speech transcript**-based unless slides are described. |

**Content category score (0–10):**  
`0.30×Org + 0.40×Expl + 0.30×Ex` with  
`Org = (0.1×SS + 0.1×LC + 0.1×CF) / 0.3`,  
`Expl = (0.2×ConceptualAccuracy + 0.1×Causal + 0.1×MultiPerspective) / 0.4`,  
`Ex = (0.1×ExampleQ + 0.1×Analogy + 0.1×Representation) / 0.3`.

---

### Delivery (0.40)

| Sub (weight of Delivery) | Criteria (weight of sub) | Definition | Reference | How MARS Works | Remarks |
|---------------------------|----------------------------|------------|-----------|----------------|---------|
| **Speech Analysis** (0.50) | **Speaking Rate** (0.10) | WPM / processing time. | Goldman-Eisler; Mayer | `calculate_metric_score` vs `SPEECH_METRICS` bands (optimal ~140–180 WPM). | Five speech criteria are **equal** within Speech Analysis; category uses `speech_components` weights internally. |
| | **Filler Word Ratio** (0.10) | Disfluencies vs total words. | Fox Tree, 2001 | Filler list + ratio → score (lower better). | |
| | **Voice Variety Index** (0.10) | Pitch/tone/emphasis variation. | Mehrabian, 1972 | Librosa-derived `voice_variety_score` → 0–10. | |
| | **Pause Effectiveness Index** (0.10) | Strategic pauses. | Rost, 2011 | Heuristic pause effectiveness score → 0–10. | |
| | **Transcription Confidence** (0.10) | ASR reliability. | Radford et al., 2022 (Whisper) | Whisper confidence mapped to 0–10. | |
| **Body Language Analysis** (0.50) | **Eye Contact** (0.10) | Visual engagement with audience/camera. | Kendon, 1967 | Vision model per frame → aggregated (temporal weighting). | **Remark (Excel):** webcasts may show instructor looking at slides/monitor — treat as limitation; consider “up” as engagement when appropriate. |
| | **Hand Gestures** (0.10) | Gestures supporting meaning. | Goldin-Meadow, 2003 | Vision scores averaged across frames. | |
| | **Posture** (0.10) | Stance and orientation. | Mehrabian, 1972 | Vision. | **Remark:** if torso not visible, score may be unreliable (camera angle). |
| | **Facial Engagement** (0.10) | Expressions, enthusiasm. | Ekman & Friesen, 1978 | Vision “engagement” channel. | **Remark:** face occluded → low/zero per product policy. |
| | **Professional Appearance** (0.10) | Attire, demeanour. | Ambady & Rosenthal, 1993 | Vision professionalism. | |

**Delivery category score (0–10):**  
`0.5 × speech_category_score + 0.5 × body_language_category_score` (each category is already 0–10 from five weighted sub-metrics).

---

### Engagement (0.40)

| Sub (weight of Engagement) | Criteria (weight of sub) | Definition | Reference | How MARS Works | Remarks |
|----------------------------|---------------------------|------------|-----------|----------------|---------|
| **Interaction Frequency** (0.40) | **Question Density** (0.40) | Instructor questions per minute. | Chi & Wylie, 2014; ICAP | Sentences with `?` + ICAP pass → QD bands → 0–10. | Maps to spreadsheet “Interaction Frequency / Question Density”. |
| **Question Quality** (0.40) | **CLI — Cognitive Level Index** (0.20) | Cognitive complexity of instructor questions. | Anderson & Krathwohl, 2001 | `(2×%Constructive + 3×%Interactive)/3` scaled to 0–10. | Within QQ block: **50%** CLI, **25%** SUI, **25%** QDS. |
| | **SUI — Student Uptake Index** (0.10) | Building on student responses. | Chin, 2007 | EQD = (Constructive+Interactive)/minute → formula `3 + 2.5×EQD` capped 0–10. | Proxy when dialogue is instructor-heavy. |
| | **QDS — Question Distribution Stability** (0.10) | Even spread of questions over time. | Dillon, 1988 | 10-bin entropy / `log2(10)` × 10. | Typo in Excel: “Stabiltiy” → **Stability**. |
| **Feedback** (0.20) | **Question Frequency** (learner) (0.10) | Frequency of learner follow-up questions. | *(none in sheet)* | LLM estimates from transcript; **confidence none** → 0. | Webcast: often **no audience audio** → 0 + remark. |
| | **Question Cognitive Level** (learner) (0.10) | Bloom-style depth of learner questions. | *(none in sheet)* | Same LLM call; paired with frequency. | Same caveat as above. |

**Engagement category score (0–10):**  
`0.40×QD + 0.40×(0.50×CLI + 0.25×SUI + 0.25×QDS) + 0.20×mean(learner_freq, learner_cognitive)`.

---

## Mapping from “old” five-way model

| Old bucket (20% each) | MARS mapping |
|------------------------|--------------|
| Speech | Inside **Delivery** (50% of Delivery) |
| Body language | Inside **Delivery** (50% of Delivery) |
| Teaching effectiveness (5 dims) | Replaced by **Content** nine criteria + legacy five kept for reporting |
| Interaction | **Engagement** minus student feedback; QD/CLI/SUI/QDS retained |
| Presentation (avg speech+visual) | **Superseded** — Delivery already combines speech + body |

The API still returns `legacy_equal_weight_overall` for comparison.

---

## Version

- **Rubric file:** `MARS Rubric v20260224.xlsx`  
- **Code version:** `MARS_RUBRIC_VERSION = "v20260224"` in `metrics_config.py`

See also: **`MARS_ALGORITHMS_AND_SCORING.md`**, **`SCORING_AND_RUBRIC_FULL.md`**.
