# MARS — Algorithms & Scoring Documentation

**Audience:** developers maintaining the pipeline; instructors and reviewers interpreting scores.  
**Version:** v20260224 (aligned with `MARS_RUBRIC_V20260224.xlsx`, sheet *Revised Rubric*).

---

## 1. What MARS outputs

| Output | Meaning | Scale |
|--------|---------|--------|
| **`overall_score`** | Primary product score | 0–10 (1 decimal) |
| **`scoring_model`** | e.g. `MARS_v20260224` | string |
| **`mars_rubric`** | Full breakdown, weights, sub-scores | object |
| **`legacy_equal_weight_overall`** | Old formula: five categories × 20% | 0–10 |

The **overall** score is **only** the MARS formula below (not the legacy five-way average).

---

## 2. Top-level formula (user-friendly)

**Overall MARS score** = **20% Content** + **40% Delivery** + **40% Engagement**

Each of **Content**, **Delivery**, and **Engagement** is first computed as a **0–10** category score. Then:

```
overall = 0.20 × Content + 0.40 × Delivery + 0.40 × Engagement
```

**Intuition:** Delivery and Engagement each contribute twice as much as Content to the final number, matching the Excel main-category weights.

---

## 3. Content (20% of overall)

### 3.1 What it measures

Instructional **explanation quality** in the transcript: organisation of ideas, accuracy and depth of explanations, and use of examples/representations.

### 3.2 Nine criteria → one number (0–10)

The LLM returns nine scores (1–10), grouped into three **sub-groups**. **Within each sub-group**, criterion weights sum to **100% of that sub-group** (as fractions of the Content block):

| Sub-group | Total weight within Content | Criteria (weights within sub-group) |
|-----------|-----------------------------|--------------------------------------|
| Content Organisation | **0.30** | Structural Sequencing **0.10** + Logical Consistency **0.10** + Closure / Framing **0.10** |
| Explanation Quality | **0.40** | **Conceptual Accuracy 0.20** + Causal Reasoning Depth **0.10** + Multi-perspective Explanation **0.10** |
| Use of examples / representation | **0.30** | Example Quality & Frequency **0.10** + Analogy / Concept Bridging **0.10** + Representation Diversity **0.10** |

**Sub-scores (each 0–10):**

- **Org** = `(0.1×SS + 0.1×LC + 0.1×CF) / 0.3` (same as mean of three).
- **Expl** = `(0.2×CA + 0.1×CR + 0.1×MP) / 0.4` — **Conceptual Accuracy counts double** (0.2 of 0.4) within Explanation Quality.
- **Ex** = `(0.1×Eq + 0.1×An + 0.1×Rd) / 0.3`.

**Then** the three sub-groups combine to **100% of the Content category**:

```
Content = 0.30 × Org + 0.40 × Expl + 0.30 × Ex
```

The **Content** category value (0–10) is then **weighted by 20%** in the overall MARS score (`0.20 × Content + 0.40 × Delivery + 0.40 × Engagement`).

### 3.3 Algorithms / data flow

1. **`analyze_pedagogy_enhanced()`** sends transcript + optional **lecture context** to the LLM; expects JSON with all nine keys plus legacy five aggregate scores.
2. **`_ensure_mars_pedagogy_fields()`** fills missing keys from legacy aggregates (`content_organization`, `communication_clarity`, `use_of_examples`).
3. **Causal reasoning depth** is **blended**:  
   `0.78 × LLM_score + 0.22 × heuristic`, where the heuristic increases with density of causal phrases (“because”, “therefore”, …) in the transcript.
4. **Code:** `compute_mars_content_category_score()` in `metrics_config.py`.

### 3.4 User caveats

- Scores reflect **what was said**, not slide design unless the instructor **describes** figures or structure.
- **Lecture context** improves alignment with intended outcomes when provided.

---

## 4. Delivery (40% of overall)

### 4.1 What it measures

**How** content is delivered: **speech** (audio metrics) and **body language** (sampled video frames).

### 4.2 Formula

```
Delivery = 0.50 × Speech_category + 0.50 × Body_language_category
```

Each of **Speech** and **Body language** is already a **0–10** score from the existing pipelines:

- **Speech** — weighted sum of: speaking rate, filler/clarity, transcription confidence, voice variety, pause effectiveness (`calculate_speech_score_enhanced`, `ANALYSIS_CONFIG["weights"]["speech_components"]`).
- **Body language** — weighted sum of: eye contact, gestures, posture, facial engagement, professionalism (`calculate_visual_score_enhanced`, `visual_components`).

### 4.3 Algorithms / data flow

- Audio: Whisper + librosa heuristics → `speech_analysis`.
- Video: sampled frames → vision LLM → per-metric scores → temporal aggregation in `analyze_visual_elements_enhanced`.

### 4.4 User caveats (from rubric)

- **Eye contact:** Webcasts often show the instructor looking at slides or monitor; scores may not match “live classroom” eye contact.
- **Posture / face:** If the frame does not show body or face, treat scores as **uncertain** (implementation may still output model scores; see remarks in `MARS_RUBRIC_V20260224.md`).

---

## 5. Engagement (40% of overall)

### 5.1 What it measures

Instructor **questioning patterns**, **higher-order dialogue proxies**, **temporal spread** of questions, and (optionally) **learner questions** from the transcript.

### 5.2 Sub-weights (within Engagement)

| Block | Weight within Engagement | Components |
|-------|---------------------------|------------|
| Interaction frequency (Question density) | 0.40 | QD → 0–10 |
| Question quality | 0.40 | CLI, SUI, QDS combined as below |
| Feedback (learner) | 0.20 | Mean of two 0–10 scores |

**Question quality block** (inside the 0.40):

```
QQ = 0.50 × CLI + 0.25 × SUI + 0.25 × QDS
```

(Excel: CLI 0.2, SUI 0.1, QDS 0.1 of Engagement → relative 2:1:1.)

**Engagement category:**

```
Engagement = 0.40 × QD + 0.40 × QQ + 0.20 × (Learner_freq + Learner_cognitive) / 2
```

### 5.3 Metric definitions (developer)

| Symbol | Name | Computation (summary) |
|--------|------|------------------------|
| **QD** | Question density | Instructor questions per minute → piecewise 0–10 (bands: ≤0.1 → 0; low/mid/high). |
| **CLI** | Cognitive Level Index | From ICAP: `(2×%Constructive + 3×%Interactive)/3` → scaled to 0–10 as `question_quality`. |
| **SUI** | Student Uptake Index | `(Constructive+Interactive)/minute` → `3 + 2.5×EQD`, capped 0–10. |
| **QDS** | Question Distribution Stability | Ten time bins; normalized entropy × 10; 0 if &lt;2 questions. |
| **Learner** | Student question frequency & cognitive | LLM estimates; if `confidence == none`, both **0** (typical webcast). |

**Code:** `analyze_interaction_engagement()`, `_compute_question_distribution_stability()`, `analyze_student_feedback_metrics()`, `compute_mars_engagement_category_score()`.

### 5.4 User caveats

- **Instructor questions** use the polished transcript and `?` detection + ICAP classification.
- **Student questions** are **inferred**; without multi-speaker audio, scores are often **0** with an explanatory remark.

---

## 6. Where to read the code

| Topic | Location |
|-------|-----------|
| MARS weights & pure functions | `metrics_config.py` — `MARS_CONFIG`, `compute_mars_*` |
| Pedagogy / nine criteria | `ai_processor.py` — `analyze_pedagogy_enhanced`, `_ensure_mars_pedagogy_fields`, `_augment_causal_reasoning_depth` |
| Student feedback | `ai_processor.py` — `analyze_student_feedback_metrics` |
| Interaction | `ai_processor.py` — `analyze_interaction_engagement` |
| Final assembly | `ai_processor.py` — `combine_analysis_enhanced` |

---

## 7. References (rubric)

Research citations appear in the Excel **Reference Sources** column (Chi, Rosenshine, Shulman, Anderson & Krathwohl, etc.). They justify **what** is measured; **how** it is measured is documented in this file and in `QUESTIONING_METRICS_BREAKDOWN.md` for questioning metrics.

---

## 8. Changelog

- **v20260224** — Revised Rubric Excel integrated: 20/40/40 main weights; nine Content criteria; Engagement includes learner-feedback block; overall score switches from five equal 20% buckets to MARS; legacy overall retained as `legacy_equal_weight_overall`.
