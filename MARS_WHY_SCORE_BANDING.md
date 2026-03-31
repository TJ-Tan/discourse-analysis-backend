# MARS “Why this score” — fixed band copy

This file documents the **predetermined one-line explanations** used in the UI for each MARS criterion, by **score band**. Bands are:

| Band   | Score range (inclusive) |
|--------|-------------------------|
| `0-2`  | 0.0 – 2.0               |
| `3-4`  | 2.1 – 4.0               |
| `5-6`  | 4.1 – 6.0               |
| `7-8`  | 6.1 – 8.0               |
| `9-10` | 8.1 – 10.0              |

Implementation: `frontend/src/marsWhyBanding.js` (must stay in sync for extraction).

---

## Content (`WHY_BANDS_CONTENT`)

### `structural_sequencing`

- **0-2:** Little clear progression; ideas feel jumped or out of order.
- **3-4:** Some structure, but the flow is uneven or hard to follow in places.
- **5-6:** Generally sensible order; main ideas usually build in a workable sequence.
- **7-8:** Strong scaffolding: foundations lead clearly toward harder ideas.
- **9-10:** Excellent sequencing—learners can follow the thread from start to finish.

### `logical_consistency`

- **0-2:** Frequent contradictions or claims that do not fit together.
- **3-4:** Several inconsistencies or unclear links between ideas.
- **5-6:** Mostly coherent; occasional gaps between what was said earlier and later.
- **7-8:** Arguments and explanations line up well across the session.
- **9-10:** Very tight logic—statements reinforce each other with few loose ends.

### `closure_framing`

- **0-2:** Openings and closings do not tie back to goals or summarise learning.
- **3-4:** Weak framing; learners may be unsure what to take away.
- **5-6:** Adequate signposting and recap in places, but not always sharp.
- **7-8:** Clear links to objectives and useful summaries of key points.
- **9-10:** Strong opening and closing—purpose and takeaways are unmistakable.

### `conceptual_accuracy`

- **0-2:** Notable inaccuracies or misleading use of key terms.
- **3-4:** Some concepts are imprecise or could confuse novices.
- **5-6:** Core ideas are mostly correct with minor slips in terminology.
- **7-8:** Concepts and relationships are explained accurately and carefully.
- **9-10:** Disciplinary content is represented very accurately and confidently.

### `causal_reasoning_depth`

- **0-2:** Almost no “why/how”; mostly labels or assertions.
- **3-4:** Limited causal explanation; reasoning feels thin.
- **5-6:** Some cause-and-effect and mechanism, mixed with surface description.
- **7-8:** Regular use of reasoning chains that explain why things happen.
- **9-10:** Deep causal explanations that make mechanisms and consequences clear.

### `multi_perspective_explanation`

- **0-2:** Single angle only; no meaningful alternatives or comparisons.
- **3-4:** Rarely steps beyond one viewpoint or example.
- **5-6:** Occasional second angles or contrasts; could go further.
- **7-8:** Several perspectives or comparisons that enrich understanding.
- **9-10:** Rich variety of angles—learners see the idea from multiple sides.

### `example_quality_frequency`

- **0-2:** Few or unhelpful examples; abstractions stay abstract.
- **3-4:** Sparse examples; quality is hit-and-miss.
- **5-6:** Enough examples to anchor ideas, with mixed depth.
- **7-8:** Well-chosen examples that clarify ideas at useful moments.
- **9-10:** Frequent, high-quality examples that make ideas concrete and memorable.

### `analogy_concept_bridging`

- **0-2:** No real bridges to prior knowledge; ideas feel isolated.
- **3-4:** Little use of analogy or familiar hooks.
- **5-6:** Some analogies or links; not always smooth or well explained.
- **7-8:** Solid use of comparisons that connect new material to what students know.
- **9-10:** Powerful analogies and bridges that speed up understanding.

### `representation_diversity`

- **0-2:** One mode only (e.g. words only); little variety.
- **3-4:** Limited variety in how ideas are represented.
- **5-6:** Mix of verbal and occasional other representations.
- **7-8:** Noticeable variety (e.g. symbols, sketches, cases) supporting the same idea.
- **9-10:** Strong multi-representation teaching that suits different learners.

---

## Delivery (`WHY_BANDS_DELIVERY`)

### `speech`

- **0-2:** Speaking is hard to follow—very fast, cluttered, or unclear much of the time.
- **3-4:** Pace or clarity is uneven; listeners may strain in several stretches.
- **5-6:** Generally understandable pace and clarity with some rough patches.
- **7-8:** Clear, well-paced delivery that most listeners can follow comfortably.
- **9-10:** Highly intelligible speech—pace, emphasis, and clarity work together very well.

### `body`

- **0-2:** Little visible engagement—eye contact, posture, or gestures weak or distracting.
- **3-4:** Non-verbal signals are inconsistent or sometimes closed-off.
- **5-6:** Adequate presence; gestures and gaze are acceptable but not standout.
- **7-8:** Open, purposeful body language that supports the message.
- **9-10:** Strong professional presence—eye contact, posture, and expression align with teaching intent.

---

## Engagement (`WHY_BANDS_ENGAGEMENT`)

### `question_density`

- **0-2:** Very few questions for the session length; limited invitation to think aloud.
- **3-4:** Low questioning rate; only occasional prompts.
- **5-6:** Moderate frequency of questions—enough to punctuate, not yet dense.
- **7-8:** Regular questioning that keeps the session intellectually active.
- **9-10:** High question density—frequent prompts without feeling chaotic.

### `cli_block`

- **0-2:** Questions stay shallow; little reasoning or dialogue-level prompting.
- **3-4:** Mostly recall-style prompts; limited higher-order questioning.
- **5-6:** Mix of recall and some deeper prompts; balance could improve.
- **7-8:** Solid share of reasoning and dialogue-oriented questions.
- **9-10:** Strong cognitive mix—many questions invite explanation, comparison, or co-construction.

### `sui`

- **0-2:** Little evidence of building on learner input (or input not audible).
- **3-4:** Rare uptake of student ideas; mostly one-way delivery.
- **5-6:** Some acknowledgement of questions or comments when they appear.
- **7-8:** Clear efforts to respond to and weave in learner contributions.
- **9-10:** Strong uptake—student ideas visibly shape the next steps of instruction.

### `qds`

- **0-2:** Questions bunched in one part of the session or almost absent.
- **3-4:** Uneven spread—large gaps without questions.
- **5-6:** Reasonable spread with a few dense or empty stretches.
- **7-8:** Questions distributed fairly across the timeline.
- **9-10:** Even, well-timed questioning across the whole session.

### `learner_question_frequency`

- **0-2:** Few or no identifiable learner questions (often a recording limit).
- **3-4:** Sparse audience questions or weak detection.
- **5-6:** Some learner questions when audio allows.
- **7-8:** Noticeable learner questioning relative to session length.
- **9-10:** Frequent learner questions or strong evidence of audience inquiry.

### `learner_question_cognitive`

- **0-2:** Learner questions look shallow or are not distinguishable.
- **3-4:** Mostly factual or brief queries from learners.
- **5-6:** Mix of simple and slightly deeper learner questions.
- **7-8:** Several questions that probe reasoning or implications.
- **9-10:** Learner questions often reach analysis, synthesis, or evaluation.
