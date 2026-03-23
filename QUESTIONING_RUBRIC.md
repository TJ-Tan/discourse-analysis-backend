# 2.4 Questioning & Interaction (20% of Overall Score)

This rubric evaluates instructor questioning and student engagement using the **ICAP framework** (Cognitive Engagement Levels). Each question is classified as **Passive**, **Active**, **Constructive**, or **Interactive**. Scores are derived from question density, cognitive level index (CLI), and effective higher-order question density.

---

## Logic and scoring algorithm

1. **Detection**  
   All sentences ending with `?` in the polished transcript are detected and matched to word timestamps.

2. **ICAP classification (LLM)**  
   Each question is classified into exactly one level:
   - **Passive:** Rhetorical, minimal demand (e.g. “Right?”, “Okay?”).
   - **Active:** Recall/factual (e.g. “What is X?”, “When did it happen?”).
   - **Constructive:** Reasoning (e.g. “Why does this happen?”, “How would you explain…?”).
   - **Interactive:** Co-construction/dialogue (e.g. “Do you agree with her? Why?”).

3. **Metrics (each scaled 1–10)**  
   - **Interaction frequency:** Questions per minute (QD). Higher density → higher score; very low density capped at 3.  
   - **Question quality:** From **Cognitive Level Index (CLI)** = (2×%Constructive + 3×%Interactive) / 3, scaled 0–100. Mapped to 1–10 as: 3 + (CLI/100)×6.5.  
   - **Student engagement opportunities:** From **effective question density (EQD)** = (Constructive + Interactive questions) per minute. Formula: 3 + EQD×2.5, capped 1–10.

4. **Overall interaction score**  
   Average of the three metrics above, rounded to one decimal.  
   **Cognitive level** label: high (CLI ≥ 50), medium (CLI ≥ 25), low (CLI &lt; 25).

5. **Output**  
   Full question list with ICAP labels is exported to Excel (#, Question, Timestamp, ICAP). ICAP counts and the three sub-scores are shown in the report.

---

## Rubric: Overall interaction score (average of Interaction Frequency, Question Quality, Student Engagement Opportunities)

| Score range | Description | Evidence | Rating | Justification | Remarks |
|-------------|-------------|----------|--------|---------------|---------|
| **9.0 – 10.0** | Socratic-style questioning; high cognitive demand; frequent higher-order (Constructive/Interactive) questions. | High question density, CLI typically ≥50, strong proportion of Constructive + Interactive; questions drive reasoning and dialogue. | Excellent | Questions drive deep thinking (Overholser, 1992). Critical thinking development; higher-order thinking improves problem-solving (Zohar & Dori, 2003). Student-centered learning; participation increases ownership (Deci & Ryan, 2000). | Target ICAP distribution: Passive &lt;10%, Active 30–50%, Constructive 25–40%, Interactive 10–20%. Context-dependent. |
| **7.5 – 8.9** | Regular questioning with good variety; mix of recall and higher-order; students have clear opportunities to engage. | Regular gestures with some variety; CLI in mid range; moderate effective question density. | Good | Regular questioning with room for depth. Good cognitive challenge with balance. Students involved but could diversify. | May still benefit from more Interactive questions and clearer wait time. |
| **6.0 – 7.4** | Questions asked but often recall-heavy; limited proportion of Constructive/Interactive; engagement opportunities present but not dominant. | Some variety; CLI in lower-mid range; effective question density modest. | Average | Questions asked but limited cognitive challenge. Some thinking required but limited depth. More instructor-centered than student-centered. | Consider increasing share of reasoning and co-construction questions. |
| **4.0 – 5.9** | Low question frequency and/or low cognitive level; few higher-order questions; limited student engagement opportunities. | Few questions per minute; low CLI; low effective question density. | Below average | Minimal questioning, low cognitive demand. Primarily surface-level thinking. Primarily passive student role. | Focus on both frequency and quality (Constructive/Interactive). |
| **1.0 – 3.9** | Very few or no questions; no meaningful higher-order engagement; no inquiry-based or student-centered structure. | No or negligible questions; CLI near zero; no effective higher-order density. | Poor | No inquiry-based learning. No critical thinking development. Students are observers only. | Introduce questioning and move toward ICAP target ranges. |

---

## Sub-metrics (used in explanations)

These three 1–10 scores are averaged to produce the overall interaction score. The same bands (9+, 7.5+, 6+, 4+, &lt;4) are used for justification text in the report.

### Question frequency (Interaction Frequency)

- **9.0+:** Socratic method. Questions drive deep thinking (Overholser, 1992).  
- **7.5–8.9:** Regular questioning with room for depth.  
- **6.0–7.4:** Questions asked but limited cognitive challenge.  
- **4.0–5.9:** Minimal questioning, low cognitive demand.  
- **&lt;4.0:** No inquiry-based learning.

### Cognitive level (Question Quality)

- **9.0+:** Critical thinking development. Higher-order thinking improves problem-solving (Zohar & Dori, 2003).  
- **7.5–8.9:** Good cognitive challenge with balance.  
- **6.0–7.4:** Some thinking required but limited depth.  
- **4.0–5.9:** Primarily surface-level thinking.  
- **&lt;4.0:** No critical thinking development.

### Interaction opportunity (Student Engagement Opportunities)

- **9.0+:** Student-centered learning. Participation increases ownership (Deci & Ryan, 2000).  
- **7.5–8.9:** Students involved but could diversify.  
- **6.0–7.4:** More instructor-centered than student-centered.  
- **4.0–5.9:** Primarily passive student role.  
- **&lt;4.0:** Students are observers only.

---

## ICAP definitions (for Excel and report)

| ICAP level    | Meaning | Example types |
|---------------|--------|----------------|
| **Passive**   | Rhetorical; minimal cognitive demand; no real answer expected. | “Right?”, “Okay?”, “See?” |
| **Active**    | Recall; factual; short or single-word answer. | “What is photosynthesis?”, “When did it happen?” |
| **Constructive** | Reasoning; requires building or explaining. | “Why does this happen?”, “How would you explain…?” |
| **Interactive**  | Co-construction; building on others’ ideas or dialogue. | “Do you agree with her? Why?”, “What would you add?” |
