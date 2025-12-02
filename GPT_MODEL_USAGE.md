# GPT-5-nano Model Usage Documentation

This document lists all locations where GPT-5-nano (previously GPT-4o-mini) is used in the MARS system and explains the purpose of each usage.

## Model Usage Locations

### 1. Transcript Post-Processing (`ai_processor.py` line 970)
**Function:** `post_process_transcript_with_gpt()`
**Purpose:** 
- Add proper punctuation (commas, periods, question marks) to raw Whisper transcript
- Segment transcript into natural sentences
- Preserve original meaning without rewriting
- Used during Step 2 (Speech Analysis) at ~42.5% progress

**Why GPT-5-nano:**
- Fast and cost-effective for text post-processing tasks
- Sufficient for punctuation and sentence segmentation
- Handles long transcripts efficiently with chunking support

---

### 2. Content Structure Analysis (`ai_processor.py` line 1195)
**Function:** `analyze_content_structure_enhanced()`
**Purpose:**
- Analyze complete lecture transcript for content organization
- Evaluate logical flow and transitions throughout the lecture
- Assess use of examples and explanations
- Identify educational techniques used
- Rate content organization, engagement techniques, communication clarity, use of examples, and knowledge checking
- Used during Step 3 (Teaching Effectiveness Analysis) at ~48% progress

**Why GPT-5-nano:**
- Efficient for analyzing full-length transcripts
- Adequate for structural and organizational analysis
- Cost-effective for comprehensive content evaluation

---

### 3. Pedagogical Analysis (`ai_processor.py` line 1456)
**Function:** `analyze_pedagogy_enhanced()`
**Purpose:**
- Comprehensive pedagogical effectiveness analysis
- Evaluate teaching effectiveness across all dimensions
- Analyze content organization, student engagement, clarity, examples usage
- Provide detailed feedback on teaching techniques
- Generate rubric-based explanations for each metric
- Used during Step 4 (Teaching Effectiveness Analysis) at ~60% progress

**Why GPT-5-nano:**
- Sufficient for pedagogical assessment tasks
- Handles complex multi-dimensional analysis
- Cost-effective for comprehensive evaluation

---

### 4. Interaction & Engagement Analysis (`ai_processor.py` line 1721)
**Function:** `analyze_interaction_engagement()`
**Purpose:**
- Analyze all detected questions (ending with ?)
- Identify high-level pedagogical questions (requiring analysis, evaluation, synthesis)
- Classify question types (high_level, clarification, rhetorical, direct, low_level)
- Calculate interaction frequency, question quality, and student engagement scores
- Determine cognitive level (low/medium/high)
- Used during Step 4.5 (Interaction & Engagement Analysis) at ~90% progress

**Why GPT-5-nano:**
- Efficient for question analysis and classification
- Adequate for pedagogical question assessment
- Cost-effective for interaction analysis

---

### 5. Comprehensive Summary Generation (`ai_processor.py` line 1978)
**Function:** `generate_comprehensive_summary()`
**Purpose:**
- Generate comprehensive evidence-based summary
- Review teaching content quality and accuracy
- Evaluate presentation effectiveness
- Assess cognitive skill development
- Provide actionable recommendations
- Extract key evidence from transcript
- Used during final summary generation

**Why GPT-5-nano:**
- Sufficient for summary generation tasks
- Handles comprehensive analysis across all dimensions
- Cost-effective for final report generation

---

### 6. PDF Summary Generation (`main.py` line 832)
**Function:** `generate_pdf_summary()`
**Purpose:**
- Generate personalized feedback paragraph (80-100 words)
- Identify strongest strength with evidence
- Identify 1-2 areas for improvement with recommendations
- Include high-level questions as evidence
- Used when exporting PDF or displaying summary on webpage

**Why GPT-5-nano:**
- Fast and efficient for personalized summary generation
- Adequate for feedback and recommendation generation
- Cost-effective for summary creation

---

## Summary

**Total GPT-5-nano Usage:** 6 locations

**All models are used for:**
- Text analysis and processing
- Educational content evaluation
- Pedagogical assessment
- Summary and feedback generation

**Note:** GPT-4o Vision (`gpt-4o`) is still used for visual frame analysis (line 1291 in `ai_processor.py`) as it requires vision capabilities that GPT-5-nano does not provide.

