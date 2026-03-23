# MARS Scoring Matrix and Rubric: Logic and Interpretation

> **Primary model (v20260224):** Overall score follows the **Revised Rubric** in `MARS_RUBRIC_V20260224.md` — **Content 20% + Delivery 40% + Engagement 40%**.  
> Full formulas and developer/user narrative: **`MARS_ALGORITHMS_AND_SCORING.md`**.

This document lists the **scoring matrix and rubric** in two ways for each part:
1. **Logic (coding)** — how the numbers are computed (formulas, inputs, code flow).
2. **Interpretation (user)** — what the scores mean for humans (rubric bands, plain-language explanation).

---

# Part A: Overall Score (MARS primary)

## Logic (coding)

- **Primary formula (MARS v20260224):**  
  `overall_score = 0.20 × mars_content + 0.40 × mars_delivery + 0.40 × mars_engagement`

- **Source:** `combine_analysis_enhanced()` in `ai_processor.py`; MARS helpers in `metrics_config.py` (`compute_mars_*`).

- **Category scores (each 0–10 before main weights):**
  - `mars_content` = `compute_mars_content_category_score(pedagogical_analysis)` — nine LLM criteria in three groups (see `MARS_ALGORITHMS_AND_SCORING.md`).
  - `mars_delivery` = `compute_mars_delivery_category_score(speech_score, visual_score)` — **50%** speech category + **50%** body language category.
  - `mars_engagement` = `compute_mars_engagement_category_score(interaction_analysis)` — QD, CLI/SUI/QDS block, learner-feedback pair.

- **Legacy reference (not used for `overall_score`):**  
  `legacy_equal_weight_overall = speech×0.20 + body×0.20 + teaching×0.20 + interaction×0.20 + presentation×0.20` — stored for comparison only.

- **Output:** `overall_score` rounded to 1 decimal; `scoring_model`: `MARS_v20260224`; `mars_rubric` object with full breakdown.

## Interpretation (user)

- Your **overall score out of 10** is **primarily** **Content** (what/how ideas are explained), **Delivery** (speech + body language combined), and **Engagement** (questioning + learner-feedback proxies).
- **Delivery** has **twice** the weight of **Content** in the final number; **Engagement** also has twice the weight of Content.
- A score of **7.0** means “good overall”; **9+** is “excellent”; **4–6** is “average to below average”; **below 4** is “needs significant improvement.”

---

# Part A2: Legacy five-way model (reference only)

The following sections (B–F) still describe **sub-metric** construction for **Speech**, **Body language**, **Teaching** (legacy five dimensions), **Interaction**, and **Presentation**. They remain accurate for **component-level** interpretation. The **overall** figure in the API is **MARS** unless you use `legacy_equal_weight_overall` for A/B testing.

---

# Part B: Speech Analysis (20%)

## Logic (coding)

- **Category score:**  
  `speech_score = rate_score × 0.25 + fluency_score × 0.25 + clarity_score × 0.20 + variety_score × 0.15 + pause_score × 0.15`  
  (weights from `ANALYSIS_CONFIG["weights"]["speech_components"]`).

- **Sub-components (each mapped to 0–10 via `calculate_metric_score` and thresholds in `SPEECH_METRICS`):**
  - **Speaking rate:** Raw = words per minute (WPM). Optimal band 140–180 → high score; linear interpolation between bands (excellent 160, good 140, average 120, poor 100 WPM).
  - **Clarity (fluency):** Raw = `filler_ratio` (filler words / total words). Lower is better (reverse_scale). Thresholds: excellent ≤2%, good ≤5%, average ≤8%, poor ≤12%.
  - **Confidence (clarity):** Raw = `confidence` (transcription confidence 0–1). Higher is better. Thresholds: excellent ≥0.95, good ≥0.90, average ≥0.85, poor ≥0.80.
  - **Voice variety:** Raw = `voice_variety_score` (0–1). Thresholds: excellent ≥0.8, good ≥0.6, average ≥0.4, poor ≥0.2.
  - **Pause effectiveness:** Raw = `pause_effectiveness_score` (0–1). Thresholds: excellent ≥0.8, good ≥0.6, average ≥0.4, poor ≥0.2.

- **Displayed “pace” in report:** `min(10, max(1, 10 - abs(WPM - 150) / 20))` (distance from 150 WPM penalises score).

## Interpretation (user)

| Metric | What it measures | Excellent (9–10) | Good (7.5–8.9) | Average (6–7.4) | Below average (4–5.9) | Poor (1–3.9) |
|--------|------------------|-------------------|----------------|-----------------|------------------------|---------------|
| **Speaking rate** | Pace (words per minute) | Optimal for comprehension (140–180 WPM); research-backed. | Slightly off but still effective. | Borderline; may lose attention or overwhelm. | Too slow or too fast; hurts learning. | Severely impairs communication. |
| **Filler ratio (clarity/fluency)** | Use of “um,” “uh,” etc. | Professional; minimal distraction. | Noticeable but acceptable. | Starts to distract. | Hurts credibility. | Very distracting; may suggest nervousness. |
| **Transcription confidence** | How clearly speech was understood (audio/clarity) | Very clear articulation. | Mostly clear. | Some unclear moments. | Often unclear. | Severely impacts understanding. |
| **Voice variety** | Variation in pitch and energy | Very dynamic; holds attention. | Moderate variation. | Some variation; can feel monotonous. | Limited; students may zone out. | Monotone; hurts engagement. |
| **Pause effectiveness** | Use of pauses for emphasis and processing | Strategic, supports processing. | Good but could be more strategic. | Some pauses; missed opportunities. | Too few (rushed) or too many (hesitant). | Disruptive or rushed. |

---

# Part C: Body Language (20%)

## Logic (coding)

- **Category score:**  
  `visual_score = eye_contact×0.25 + gestures×0.20 + posture×0.20 + engagement×0.20 + professionalism×0.15`  
  (weights from `ANALYSIS_CONFIG["weights"]["visual_components"]`).

- **Inputs:** All five come from `visual_analysis['scores']`, which is populated by the **Vision (GPT) frame analysis**: each sampled frame is scored 1–10 for eye contact, gestures, posture, engagement, professionalism; per-metric scores are averaged (or aggregated) across frames.

- **Source:** `analyze_visual_elements_enhanced()` → model returns scores per frame → aggregated into `scores` dict.

## Interpretation (user)

| Metric | What it measures | Excellent (9–10) | Good (7.5–8.9) | Average (6–7.4) | Below average (4–5.9) | Poor (1–3.9) |
|--------|------------------|-------------------|----------------|-----------------|------------------------|---------------|
| **Eye contact** | Connection with audience/camera | Builds connection and trust; supports engagement. | Strong with acceptable use of notes. | Connection present but weakened by looking away. | Weak; seems disengaged or over-reliant on notes. | Little/no connection; reading or distracted. |
| **Gestures** | Hand gestures supporting content | Supports comprehension; can improve retention. | Helpful but could be more intentional. | Some visual interest; limited impact. | Distracting or no support. | Actively harms communication. |
| **Posture** | Stance and presence | Projects authority and confidence. | Professional with minor room to improve. | Adequate but could show more confidence. | Undermines authority. | Clearly impacts perceived competence. |
| **Facial engagement** | Expressiveness and energy | Conveys enthusiasm; supports engagement. | Engaging with some flat moments. | Adequate; could show more energy. | Little enthusiasm or inappropriate emotion. | Disconnects students from content. |
| **Professionalism** | Overall professional appearance | Supports credibility. | Appropriate for context. | Slightly reduces perceived authority. | Detracts from professional image. | Seriously undermines credibility. |

*Note: Visual analysis depends on recording quality and framing (e.g. face visible, camera angle).*

---

# Part D: Teaching Effectiveness (20%)

## Logic (coding)

- **Category score:**  
  `pedagogy_score = content_organization×0.25 + engagement_techniques×0.20 + communication_clarity×0.20 + use_of_examples×0.20 + knowledge_checking×0.15`  
  (weights from `ANALYSIS_CONFIG["weights"]["pedagogy_components"]`).

- **Inputs:** All five are **LLM-derived** (GPT) from the full transcript and any structure/context: the pedagogical analysis step returns numeric scores (typically 0–10 or similar) for content_organization, engagement_techniques, communication_clarity, use_of_examples, knowledge_checking. These are read from `pedagogical_analysis` and combined with the weights above.

## Interpretation (user)

| Metric | What it measures | Excellent (9–10) | Good (7.5–8.9) | Average (6–7.4) | Below average (4–5.9) | Poor (1–3.9) |
|--------|------------------|-------------------|----------------|-----------------|------------------------|---------------|
| **Content organisation** | Structure and flow of content | Reduces cognitive load; clear structure. | Well organised with room to refine. | Followable but connections may be missed. | Hard to build a clear mental model. | Severely impairs learning. |
| **Engagement techniques** | Use of active learning / interaction | Strong evidence of active learning benefits. | Engages but could diversify. | Mostly lecture with limited interaction. | Largely passive. | Students only passive recipients. |
| **Communication clarity** | How clearly ideas are explained | Very comprehensible. | Clear with room for precision. | Main ideas clear; details missed. | Students struggle with concepts. | Students cannot follow. |
| **Use of examples** | Concrete examples and applications | Examples support transfer of learning. | Examples present but could vary more. | Basic examples. | Concepts stay abstract. | Students cannot relate to content. |
| **Knowledge checking** | Checking understanding (e.g. questions, tasks) | Formative checking supports learning. | Some checking; could be more frequent. | Assumes understanding. | May miss confusion. | No real feedback loop. |

---

# Part E: Interaction & Engagement (20%)

## Logic (coding)

- **Category score:**  
  `interaction_score = (interaction_frequency + question_quality + student_uptake_index + question_distribution_stability) / 4`  
  (average of four 0–10 components). Also reported as **percentages** (each component × 10 → 0–100%; overall = average of four percentages).

- **Sub-components:**
  1. **Interaction frequency (0–10):** QD = total_questions / duration_minutes. QD ≤ 0.1 → 0; 0.1 < QD < 0.5 → 1–3; 0.5 ≤ QD < 1.5 → 4–7; QD ≥ 1.5 → 8–10 (piecewise linear).
  2. **Question quality (0–10):** CLI_100 = (2×%Constructive + 3×%Interactive)/3 × 100; question_quality = CLI_100/10. (ICAP: each question classified Passive/Active/Constructive/Interactive by LLM.)
  3. **Student Uptake Index – SUI (0–10):** EQD = (count_constructive + count_interactive) / duration_minutes; SUI = 3 + EQD×2.5, clamped 0–10.
  4. **Question Distribution Stability – QDS (0–10):** Lecture split into 10 time bins; entropy of question counts across bins; normalized by log2(10); ×10 → 0–10. (0 if &lt;2 questions.)

- **Data flow:** Transcript → detect sentences ending with `?` → LLM assigns ICAP per question → counts and timestamps → QD, CLI, EQD, QDS → four scores → average → interaction_score.

- **Detailed formulas:** See `QUESTIONING_METRICS_BREAKDOWN.md`.

## Interpretation (user)

| Metric | What it measures | Excellent (9–10) | Good (7.5–8.9) | Average (6–7.4) | Below average (4–5.9) | Poor (1–3.9) |
|--------|------------------|-------------------|----------------|-----------------|------------------------|---------------|
| **Question frequency** | How often you ask questions | Socratic-style; supports deep thinking. | Regular questioning with room for depth. | Some questions; limited cognitive challenge. | Few questions; low demand. | No inquiry-based learning. |
| **Question quality (cognitive level)** | How much questions require thinking | Develops critical thinking; higher-order. | Good cognitive challenge and balance. | Some thinking; limited depth. | Mostly surface-level. | No real critical thinking development. |
| **Student Uptake Index (SUI)** | Higher-order questions (reasoning/dialogue) per minute | Student-centered; participation encouraged. | Students involved; could diversify. | More instructor-centered. | Largely passive student role. | Students only observers. |
| **Question Distribution Stability (QDS)** | Spread of questions over time | Questions spread across the session; sustained engagement. | Good spread with minor clustering. | Some spread; tendency to cluster. | Questions in a limited part of session. | Clustered in one segment or too few to tell. |

- **ICAP (for reference):** Questions are labelled **Passive** (rhetorical), **Active** (recall), **Constructive** (reasoning), **Interactive** (dialogue/co-construction). More Constructive and Interactive → higher question quality and SUI. Percentages for each component (0–100%) show your share of the 20% interaction category.

---

# Part F: Presentation Skills (20%)

## Logic (coding)

- **Category score:**  
  `presentation_score = (speech_score + visual_score) / 2`  
  (average of Speech and Body language category scores).

- **Reported sub-metrics (for explanation only; not a separate weighted formula):**
  - **Energy:** Raw = `speaking_ratio` (proportion of time speaking, 0–1). Score = raw × 10, clamped 1–10.
  - **Voice modulation:** Same source as Speech “voice variety” (voice_variety_score × 10).
  - **Professionalism:** Same as Body language “professionalism” from visual scores.
  - **Time management:** `(speaking_ratio × 0.7 + pause_effectiveness_score × 0.3) × 10`, clamped 1–10.

## Interpretation (user)

| Metric | What it measures | Excellent (9–10) | Good (7.5–8.9) | Average (6–7.4) | Below average (4–5.9) | Poor (1–3.9) |
|--------|------------------|-------------------|----------------|-----------------|------------------------|---------------|
| **Energy** | How much of the time you’re speaking / presence | Enthusiasm comes across; supports engagement. | Engaging with room for consistency. | Functional but not very motivating. | Does not inspire or engage. | Disengages students. |
| **Voice modulation** | Variation in delivery | Prosodic emphasis aids comprehension. | Engaging vocal delivery. | Some variety; risk of monotony. | Approaching monotone. | Flat, disengaging. |
| **Professionalism** | As in Body language | Credibility and expertise signalled. | Appropriate. | Slight impact on authority. | Weakens professional image. | Undermines credibility. |
| **Time management** | Pacing and use of pauses | Pacing supports cognitive load. | Effective with minor improvements. | Needs attention. | Affects coverage and understanding. | Severely disrupts learning. |

- **Overall presentation:** Your Presentation score is the average of your **Speech** and **Body language** scores, so improving either improves Presentation as well.

---

# Part G: Rubric Bands (All Metrics)

## Logic (coding)

- **Band boundaries** (used in `get_rubric_explanation()`):  
  - **Excellent:** score ≥ 9.0  
  - **Good:** score ≥ 7.5 and &lt; 9.0  
  - **Average:** score ≥ 6.0 and &lt; 7.5  
  - **Below average:** score ≥ 4.0 and &lt; 6.0  
  - **Poor:** score &lt; 4.0  

- **Returned for each metric:** `rating` (label), `justification` (research/explanation text), `remarks` (caveats), `score_range` (e.g. "7.5-8.9").

- **Where used:** Report “explanations” for each category pull from `get_rubric_explanation(metric_name, value, score)` so the same bands and text drive the user-facing justification everywhere.

## Interpretation (user)

- Every metric is given a **verbal rating** (Excellent / Good / Average / Below average / Poor) and a **short justification** (often with a research or practice note). **Remarks** add context (e.g. culture, subject, recording limits). Use these to understand not only the number but *why* it was assessed that way and what to work on.

---

# Quick Reference: Where Things Live

| Item | Logic (code) | Interpretation (user) |
|------|----------------|------------------------|
| **Overall score (primary)** | `combine_analysis_enhanced`; MARS `0.20×C + 0.40×D + 0.40×E` | Content, Delivery, Engagement — see `MARS_ALGORITHMS_AND_SCORING.md`. |
| **Legacy overall** | `legacy_equal_weight_overall` in API payload | Old five × 20% formula for comparison. |
| **Speech** | `calculate_speech_score_enhanced`; 5 components from `calculate_metric_score` + SPEECH_METRICS | Half of **Delivery** (with body language); Part B. |
| **Body language** | `calculate_visual_score_enhanced`; 5 scores from Vision API | Half of **Delivery**; Part C. |
| **Content (MARS)** | Nine LLM criteria → `compute_mars_content_category_score` | Part D legacy text + `MARS_RUBRIC_V20260224.md`. |
| **Teaching (legacy 5)** | `calculate_pedagogy_score_enhanced` | Still reported under teaching_effectiveness for detail. |
| **Interaction & engagement** | `analyze_interaction_engagement` + student feedback LLM; QD, CLI, SUI, QDS | **Engagement** category; `QUESTIONING_METRICS_BREAKDOWN.md`. |
| **Presentation** | (Speech + Body) / 2 in legacy only | For MARS, use **Delivery** instead. |
| **Rubric bands** | `get_rubric_explanation(metric_name, value, score)`; 9+, 7.5+, 6+, 4+, &lt;4 | Excellent / Good / Average / Below average / Poor + justification + remarks. |

This document, **`MARS_RUBRIC_V20260224.md`**, **`MARS_ALGORITHMS_AND_SCORING.md`**, and `QUESTIONING_METRICS_BREAKDOWN.md` together document the full matrix for coding and user terms.
