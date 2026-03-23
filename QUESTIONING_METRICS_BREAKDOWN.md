# Question & Interaction Metrics: How Each Number Is Computed

This document breaks down **ICAP**, **Interaction frequency (QD)**, **Question quality**, **Student Uptake Index (SUI)**, **Question Distribution Stability (QDS)**, **Cognitive level**, and the **Overall interaction score** (average of four components)—and shows exactly how each value is obtained. All component scores are 0–10 and also reported as **percentages (0–100)**.

> **MARS v20260224:** These four metrics feed the **Engagement** main category together with **learner-question feedback** scores. The **final course overall** uses MARS (`0.20×Content + 0.40×Delivery + 0.40×Engagement`), not the old five equal 20% buckets — see **`MARS_ALGORITHMS_AND_SCORING.md`**. The **average of four** interaction components below is still computed as `interaction_analysis['score']` for reporting and is one input to the Engagement category formula.

---

## 1. ICAP (the four levels)

**What it is:** Each question is classified into exactly one of four cognitive-engagement levels (ICAP framework).

| Level | Meaning | How we get it |
|-------|--------|----------------|
| **Passive** | Rhetorical; minimal demand; no real answer expected | LLM (GPT) classifies each question; count = number of questions with `icap === "Passive"`. |
| **Active** | Recall / factual; short or single-word answer | Count = number with `icap === "Active"`. |
| **Constructive** | Reasoning; requires building or explaining | Count = number with `icap === "Constructive"`. |
| **Interactive** | Co-construction; building on others’ ideas | Count = number with `icap === "Interactive"`. |

**How the counts are obtained:**

1. **List of questions**  
   From the polished transcript, every sentence that **ends with `?`**, starts with a capital letter, and has at least 3 words is taken as a question. Word timestamps give the start time for each.

2. **ICAP label per question**  
   The full list is sent to the LLM (e.g. gpt-5-nano) with the ICAP definitions. The model returns for each question one of: `"Passive"`, `"Active"`, `"Constructive"`, `"Interactive"`.

3. **Counts**  
   - If the model returns `count_passive`, `count_active`, `count_constructive`, `count_interactive`, those are used.  
   - If the counts don’t add up to `total_questions`, we recompute by counting how many items in `all_questions_analyzed` have each `icap` value.

**Formulas (counts):**

- `count_passive`   = number of questions where `icap === "Passive"`
- `count_active`    = number of questions where `icap === "Active"`
- `count_constructive` = number of questions where `icap === "Constructive"`
- `count_interactive`  = number of questions where `icap === "Interactive"`
- `total_questions` = `count_passive + count_active + count_constructive + count_interactive`

**Percentages (for interpretation only, not used directly in scores):**

- `% Passive`   = `count_passive   / total_questions`
- `% Active`    = `count_active    / total_questions`
- `% Constructive` = `count_constructive / total_questions`
- `% Interactive`  = `count_interactive  / total_questions`

---

## 2. Interaction frequency (0–10, reported as 0–100%)

**What it is:** Score for question density (questions per minute). **High density → 8–10; mid → 4–7; low → 1–3; no density (QD 0–0.1) → 0.**

**How we get the number:**

1. **Duration**  
   `duration_minutes` = speech length in minutes (≥ 0.1).

2. **Question density**  
   `QD = total_questions / duration_minutes`

3. **Map QD → 0–10** (piecewise):

   | QD range | Score range | Formula |
   |----------|-------------|---------|
   | QD ≤ 0.1 | **0** (no density) | 0 |
   | 0.1 < QD < 0.5 | **1–3** (low) | 1 + (QD - 0.1) / 0.4 × 2 |
   | 0.5 ≤ QD < 1.5 | **4–7** (mid) | 4 + (QD - 0.5) / 1.0 × 3 |
   | QD ≥ 1.5 | **8–10** (high) | 8 + min(2, (QD - 1.5) × 2) |

4. **Percentage**  
   `interaction_frequency_pct = interaction_frequency × 10` (0–100).

**If there are no questions:**  
`interaction_frequency = 0`, `interaction_frequency_pct = 0`.

---

## 3. Question quality (0–10, reported as 0–100%)

**What it is:** A 1–10 score for the cognitive level of questions (share of Constructive and Interactive).

**How we get the number:**

1. **Percentages** (as above)  
   `pct_constructive` = `count_constructive / total_questions`  
   `pct_interactive`  = `count_interactive  / total_questions`

2. **Cognitive Level Index (CLI), 0–100**  
   `CLI_raw` = `(2 × pct_constructive + 3 × pct_interactive) / 3`  
   - Weights: Interactive (×3) > Constructive (×2).  
   - Range: 0 (no C or I) to 1 (all Interactive).  
   `CLI_100` = `CLI_raw × 100`  → **0–100**.

3. **Map CLI 0–100 → 0–10**  
   `question_quality` = `(CLI_100 / 100) × 10`.  
   - CLI = 0   → 0  
   - CLI = 100 → 10  

4. **Percentage**  
   `question_quality_pct = question_quality × 10` (0–100).

**If there are no questions:**  
`question_quality = 0`, `question_quality_pct = 0`.

**Reported “Cognitive level index”** in the API is the same **CLI_100** (or 0 if no questions).

---

## 4. Student Uptake Index – SUI (0–10, reported as 0–100%)

**What it is:** Score for higher-order question density (Constructive + Interactive per minute). *Renamed from “Student engagement opportunities”.*

**How we get the number:**

1. **Effective question density**  
   `EQD = (count_constructive + count_interactive) / duration_minutes`

2. **Map EQD → 0–10**  
   `student_uptake_index` = `3 + EQD × 2.5`, clamped to **[0, 10]** and rounded.

3. **Percentage**  
   `student_uptake_index_pct = student_uptake_index × 10` (0–100).

**If there are no questions:**  
`student_uptake_index = 0`, `student_uptake_index_pct = 0`.

---

## 5. Question Distribution Stability – QDS (0–10, reported as 0–100%)

**What it is:** How spread out questions are over the lecture (e.g. at 3 min, 18 min, 38 min in a 50 min lecture = good spread).

**How we get the number:**

1. **Time bins**  
   Lecture timeline is split into **10 equal bins**. For each question, `start_time` (seconds) determines its bin.

2. **Entropy**  
   Count questions per bin; compute proportions p_i.  
   Entropy `H = -Σ p_i log2(p_i)`. Maximum when uniform (1 question per bin) = log2(10).

3. **Normalized score 0–10**  
   `normalized = H / log2(10)` (0–1).  
   `question_distribution_stability = normalized × 10`, clamped to **[0, 10]**.

4. **Percentage**  
   `question_distribution_stability_pct = question_distribution_stability × 10` (0–100).

**If there are 0 or 1 question:**  
`question_distribution_stability = 0`.

---

## 6. Cognitive level (label: low / medium / high)

**What it is:** A text label for overall question cognitive level, from CLI.

**How we get it:**

- Use **CLI_100** as above (0–100).
- **If no questions:** `cognitive_level = "low"`.
- **If there are questions:**
  - `CLI_100 ≥ 50` → **"high"**
  - `CLI_100 ≥ 25` and &lt; 50 → **"medium"**
  - `CLI_100 < 25` → **"low"**

So the **number** behind the label is **CLI_100** (the same as “Cognitive level index”).

---

## 7. Overall interaction score (0–10 and 0–100%) – 20% sub-category

**What it is:** The “Interaction & Engagement” score that **contributes 20%** to the overall grade. It is the average of the **four** components (each 0–10), then expressed as a percentage.

**How we get the number:**

- **Score (0–10):**  
  `score = (interaction_frequency + question_quality + student_uptake_index + question_distribution_stability) / 4`  
  Rounded to 1 decimal.

- **Overall percentage (0–100):**  
  `overall_interaction_pct = score × 10`  
  (or equivalently the average of the four _pct values).

**Contribution to overall:**  
The overall course score uses `interaction_score × 0.20` (20% weight). So each of the four sub-metrics contributes 25% of this 20% category (i.e. 5% of the total each when equal-weighted).

---

## 8. Quick reference: inputs and outputs

| Output | Depends on | Formula / rule |
|--------|------------|------------------|
| **ICAP counts** | Transcript + LLM | Count of questions per ICAP label. |
| **Total questions** | Transcript | Sentences ending with `?` (capital, ≥3 words); deduped. |
| **Interaction frequency (0–10)** | `total_questions`, `duration_minutes` | QD = total/min; QD ≤0.1→0, 0.1–0.5→1–3, 0.5–1.5→4–7, ≥1.5→8–10. |
| **Interaction frequency %** | Above | × 10 (0–100). |
| **Question quality (0–10)** | CLI from C, I, total | CLI_100 = (2×%C + 3×%I)/3 × 100; quality = CLI_100/10. |
| **Question quality %** | Above | × 10 (0–100). |
| **Student Uptake Index – SUI (0–10)** | C, I, duration | EQD = (C+I)/min; 3 + EQD×2.5, clamp 0–10. |
| **SUI %** | Above | × 10 (0–100). |
| **Question Distribution Stability – QDS (0–10)** | Question `start_time`s, duration | 10 bins, entropy / log2(10) × 10. |
| **QDS %** | Above | × 10 (0–100). |
| **Cognitive level** | CLI_100 | high ≥50, medium ≥25, low &lt;25. |
| **Overall interaction score (0–10)** | Four components | Average of the four 0–10 scores. |
| **Overall interaction %** | Above | × 10 (0–100); contributes 20% to total. |
| **Cognitive level index** | Same as CLI | (2×C + 3×I)/3 / total_questions × 100 (or 0 if no questions). |

---

## 9. Example (numbers only)

- Lecture length: **10 minutes**.  
- Questions: **20** total → **Passive 2, Active 10, Constructive 5, Interactive 3**.

Then:

- **QD** = 20/10 = **2.0** → interaction_frequency = 8 + min(2, (2−1.5)×2) = **9.0** (high density); **90%**.
- **CLI_100** ≈ **31.7** → question_quality = 3.17 → **3.2** (0–10); **32%**.
- **EQD** = (5+3)/10 = **0.8** → student_uptake_index = 3 + 0.8×2.5 = **5.0**; **50%**.
- **QDS** = from 20 question timestamps in 10 bins → assume good spread → e.g. **7.0**; **70%**.
- **Cognitive level** = 31.7 ≥ 25 → **"medium"**.
- **Overall** = (9 + 3.2 + 5 + 7) / 4 = **6.05** → **60.5%** (contributes 20% of total score).

So you get: **how many questions**, **how they’re labeled (ICAP)**, **how often they’re asked (frequency)**, **how “deep” they are (quality)**, **how many higher-order per minute (engagement)**, and **one combined score plus a cognitive-level label**.
