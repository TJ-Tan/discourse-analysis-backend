# Whisper Transcription and Question Detection Algorithm

## Overview

The MARS system uses OpenAI's Whisper API for audio transcription and implements a hybrid approach for question detection that combines automatic punctuation, intonation analysis, pattern matching, and AI-powered semantic analysis.

---

## 1. Whisper Transcription Process

### 1.1 Audio Processing Pipeline

The system handles audio transcription through multiple pathways depending on file size:

#### **Small Files (< 20MB)**
- **Direct Processing**: Audio is sent directly to Whisper API
- **Format**: 16kHz mono WAV (PCM)
- **Response Format**: `verbose_json` with word-level timestamps

#### **Large Files (> 20MB)**
- **Chunked Processing**: Audio is split into manageable chunks
- **Chunk Size**: ~10 minutes per chunk (600 seconds)
- **Overlap**: 30 seconds between chunks to prevent word cutting
- **Method**: Uses FFmpeg for memory-efficient streaming

### 1.2 Whisper API Configuration

```python
openai_client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["word"],
    language="en",  # Force English transcription
    prompt="This is a lecture transcript. Please add proper punctuation including commas, periods, question marks, and exclamation marks. Use question marks for questions based on intonation and sentence structure."
)
```

**Key Parameters:**
- **`language="en"`**: Forces English transcription to prevent language detection errors
- **`prompt`**: Encourages Whisper to add proper punctuation automatically
- **`timestamp_granularities=["word"]`**: Provides word-level timestamps for precise analysis
- **`verbose_json`**: Returns detailed response with text, words array, and metadata

### 1.3 Chunked Transcription Process

For large files, the system:

1. **Calculates chunk parameters**:
   - Target duration: 600 seconds (10 minutes)
   - Overlap: 30 seconds
   - Total chunks: `ceil(video_duration / 600)`

2. **Extracts audio chunks using FFmpeg**:
   ```bash
   ffmpeg -i input.mp4 -ss START_TIME -t DURATION \
          -acodec pcm_s16le -ar 16000 -ac 1 output.wav
   ```

3. **Transcribes each chunk** with the same Whisper configuration

4. **Adjusts timestamps** to global video time:
   ```python
   for word_data in chunk_words:
       word_data['start'] += start_time
       word_data['end'] += start_time
   ```

5. **Combines transcripts**:
   - Merges all chunk texts into a single transcript
   - Combines all word timestamps into a unified array
   - Returns a `CombinedResponse` object with `.text` and `.words` attributes

### 1.4 Output Structure

The transcription returns:
- **`text`**: Full transcript with punctuation (from Whisper's automatic punctuation)
- **`words`**: Array of word objects with:
  - `word`: The transcribed word
  - `start`: Start timestamp (seconds)
  - `end`: End timestamp (seconds)

---

## 2. Question Detection Algorithm

The system uses a **four-stage hybrid approach** to detect questions:

### Stage 1: Automatic Punctuation (Whisper)

**How it works:**
- Whisper API automatically adds punctuation based on:
  - Audio pauses and intonation patterns
  - Sentence structure and context
  - The provided prompt encouraging question marks

**Output:**
- Transcript with punctuation marks (`.`, `?`, `!`, `,`)
- Question marks (`?`) indicate potential questions

**Limitations:**
- Whisper may miss some questions, especially if:
  - Intonation is subtle
  - No clear pause before the question
  - Question structure is implicit

### Stage 2: Intonation Analysis (Pitch Detection)

**Algorithm:**
1. **Extract pitch track** using librosa's `piptrack`:
   ```python
   pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, threshold=0.1)
   ```

2. **Identify sentence boundaries**:
   - Look for punctuation in transcript
   - Detect pauses > 0.8 seconds between words
   - Map to word timestamps

3. **Analyze pitch at sentence endings**:
   - Extract pitch values in the last 0.5 seconds before sentence end
   - Split window into first half and second half
   - Calculate pitch rise ratio: `(second_half - first_half) / first_half`

4. **Detect rising intonation**:
   - **Threshold**: 15% pitch rise indicates a question
   - **Confidence**: Normalized to 0-1 scale based on rise magnitude
   - **Minimum confidence**: 0.5 (50%) to add question mark

5. **Enhance transcript**:
   - Add question marks where rising intonation is detected
   - Only if punctuation is not already present
   - Map to closest word timestamp

**Code Flow:**
```python
def analyze_intonation_for_questions(audio_data, sample_rate, word_timestamps):
    # Extract pitch track
    pitches, magnitudes = librosa.piptrack(...)
    
    # Find sentence boundaries
    sentence_end_timestamps = [...]
    
    # Analyze pitch at each sentence end
    for sentence_end_time in sentence_end_timestamps:
        window_pitches = extract_pitches_in_window(...)
        pitch_rise_ratio = calculate_rise(window_pitches)
        
        if pitch_rise_ratio > 0.15:  # 15% rise threshold
            question_timestamps.append({
                'timestamp': sentence_end_time,
                'pitch_rise': pitch_rise_ratio,
                'confidence': normalize(pitch_rise_ratio)
            })
    
    return {'question_timestamps': question_timestamps}
```

**Advantages:**
- Detects questions even without explicit question words
- Works for implicit questions (e.g., "You understand?")
- Complements Whisper's punctuation

**Limitations:**
- Requires clear audio quality
- May miss questions with flat intonation
- Needs sufficient pitch variation

### Stage 3: Pattern Matching

**Algorithm:**
1. **Question starter words**:
   ```python
   question_starters = [
       'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom',
       'can', 'could', 'would', 'should', 'will', 'may', 'might', 'must',
       'do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had'
   ]
   ```

2. **Question patterns** (2-3 word windows):
   ```python
   question_patterns = [
       'what is', 'what are', 'what do', 'what does', 'what did',
       'how do', 'how does', 'how can', 'how would',
       'why do', 'why does', 'why is', 'why are',
       'can you', 'could you', 'would you', 'should we',
       'do you', 'does it', 'did you', 'is it', 'are you',
       'let\'s', 'turn to', 'think about', 'consider'
   ]
   ```

3. **Detection process**:
   - Scan word timestamps sequentially
   - Check if word starts with question starter
   - Check 2-word and 3-word patterns
   - Capture full sentence (up to 20 words)
   - Assign confidence: 'high' for patterns, 'medium' for starters

**Output:**
- List of detected questions with:
  - Question text
  - Start timestamp
  - Confidence level
  - Detection method

### Stage 4: AI-Powered Semantic Analysis (GPT-4o-mini)

**Purpose:**
- Validate pattern-matched questions
- Detect implicit questions missed by patterns
- Analyze semantic meaning and intent

**Process:**
1. **Input preparation**:
   - Full transcript (up to 6000 characters)
   - Pattern-matched questions as hints
   - Explicit instruction about missing punctuation

2. **AI Prompt**:
   ```
   IMPORTANT: The transcript has NO punctuation marks (no question marks, periods, etc.). 
   You must identify questions based on:
   1. Sentence structure (question words: what, why, how, when, where, who, which, can, could, would, should, do, does, did, is, are, was, were, have, has)
   2. Semantic meaning and intent (even if not explicitly structured as questions)
   3. Interaction patterns (e.g., "Let's hear from...", "Can someone...", "Turn to your partner...")
   4. Cognitive engagement prompts (e.g., "Analyze...", "Compare...", "Evaluate...", "Think about...")
   
   Pattern-matched potential questions (for reference):
   [List of pattern-matched questions]
   ```

3. **AI Analysis**:
   - Model: GPT-4o-mini
   - Temperature: 0.3 (for consistency)
   - Max tokens: 2000
   - Returns JSON with:
     - `total_questions`: Count
     - `questions`: Array of question objects
     - `interaction_moments`: Other engagement moments
     - Scores for interaction frequency, question quality, etc.

4. **Result Merging**:
   - Combine AI-detected questions with pattern-matched ones
   - Remove duplicates
   - Map to precise timestamps using word matching

**Code Flow:**
```python
def analyze_interaction_engagement(transcript, words_data):
    # Step 1: Pattern matching
    pattern_questions = detect_questions_pattern_matching(words_data)
    
    # Step 2: AI analysis with pattern hints
    ai_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        temperature=0.3,
        max_tokens=2000
    )
    
    # Step 3: Merge results
    all_questions = merge_ai_and_pattern_questions(...)
    
    # Step 4: Map to timestamps
    for question in all_questions:
        question['precise_timestamp'] = find_word_timestamp(...)
    
    return all_questions
```

---

## 3. Integration and Final Output

### 3.1 Transcript Enhancement Flow

```
1. Whisper Transcription
   ↓
   Transcript with automatic punctuation
   ↓
2. Intonation Analysis
   ↓
   Detect rising pitch → Add question marks
   ↓
3. Pattern Matching
   ↓
   Detect question words/patterns
   ↓
4. AI Semantic Analysis
   ↓
   Validate and detect implicit questions
   ↓
5. Final Transcript
   - Enhanced with intonation-based punctuation
   - Question markers from all stages
   - Word-level timestamps
```

### 3.2 Question Detection Scoring

The system calculates:
- **Total Questions**: Count from all detection methods
- **Interaction Frequency**: Based on questions per minute
- **Question Quality**: Assessed by AI (cognitive level, clarity)
- **Student Engagement Opportunities**: Count of interaction moments

**Scoring Logic:**
- If `total_questions == 0` and `total_interactions == 0`:
  - All scores set to 3.0/10 (low)
  - Cognitive level: "low"
- Otherwise:
  - Scores calculated from AI analysis
  - Cognitive level: "low", "medium", or "high"

---

## 4. Technical Details

### 4.1 Libraries Used

- **`librosa`**: Audio analysis, pitch extraction (`piptrack`)
- **`openai`**: Whisper API and GPT-4o-mini
- **`ffmpeg`**: Audio chunking for large files
- **`numpy`**: Numerical operations for pitch analysis

### 4.2 Performance Optimizations

1. **Memory Efficiency**:
   - Streaming FFmpeg for large files
   - Chunked processing to stay under 25MB Whisper limit
   - Garbage collection between chunks

2. **Error Handling**:
   - Retry logic for API calls (exponential backoff)
   - Fallback to librosa method if streaming fails
   - Graceful degradation if intonation analysis fails

3. **Caching**:
   - Word timestamps cached for multiple analyses
   - Transcript text reused across different analysis stages

---

## 5. Limitations and Future Improvements

### Current Limitations

1. **Intonation Analysis**:
   - Requires clear audio quality
   - May miss questions with flat intonation
   - Sensitive to background noise

2. **Pattern Matching**:
   - May miss non-standard question structures
   - Limited to English question patterns

3. **AI Analysis**:
   - Token limits restrict full transcript analysis
   - May miss very implicit questions

### Potential Improvements

1. **Enhanced Intonation**:
   - Use more sophisticated pitch tracking algorithms
   - Analyze energy patterns in addition to pitch
   - Consider speaker-specific intonation patterns

2. **Better Pattern Matching**:
   - Expand question pattern database
   - Use machine learning for pattern recognition
   - Support multiple languages

3. **AI Enhancements**:
   - Use GPT-4 for better semantic understanding
   - Implement chunked transcript analysis for very long videos
   - Add context-aware question detection

---

## 6. Example Workflow

**Input**: 30-minute lecture video

1. **Audio Extraction**: Extract audio to 16kHz mono WAV
2. **File Size Check**: 45MB → Requires chunking
3. **Chunking**: Split into 3 chunks (10 min each, 30s overlap)
4. **Whisper Transcription**:
   - Chunk 1: "Today we'll discuss... What is power?"
   - Chunk 2: "Let's think about... How does this apply?"
   - Chunk 3: "Any questions? Turn to your partner..."
5. **Intonation Analysis**: Detects rising pitch at "What is power?" and "Any questions?"
6. **Pattern Matching**: Finds "What is", "How does", "Any questions"
7. **AI Analysis**: Validates all, adds "Let's think about" as implicit question
8. **Final Output**: 
   - Transcript with proper punctuation
   - 4 questions detected
   - Interaction frequency: 8.0/10
   - Question quality: 7.5/10

---

## References

- [OpenAI Whisper API Documentation](https://platform.openai.com/docs/guides/speech-to-text)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

---

**Last Updated**: November 2024  
**Version**: 3.0.0  
**Maintained by**: MARS Development Team


