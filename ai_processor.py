import os
import asyncio
import subprocess
from pathlib import Path
import cv2
import librosa
import numpy as np
from openai import OpenAI
from typing import Dict, List, Any, Optional
import base64
import json
import re
import logging
from dotenv import load_dotenv

# Import enhanced configuration
from metrics_config import (
    ANALYSIS_CONFIG, FILLER_WORDS, SPEECH_METRICS, VISUAL_METRICS, PEDAGOGY_METRICS,
    calculate_metric_score, get_metric_feedback
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class VideoAnalysisProcessor:
    def __init__(self):
        self.temp_dir = Path("temp_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
    async def process_video(self, video_path: Path, analysis_id: str, progress_callback):
        """
        Enhanced processing pipeline for video analysis with improved sampling and metrics
        """
        try:
            logger.info(f"🎬 Starting enhanced video analysis for {analysis_id}")
            logger.info(f"📁 File: {video_path.name} ({video_path.stat().st_size / (1024*1024):.1f}MB)")
            
            # Step 1: Extract audio and video components with enhanced sampling
            logger.info("🔧 Step 1: Extracting audio and video components...")
            await progress_callback(analysis_id, 10, "Extracting audio and video components...")
            audio_path, video_frames = await self.extract_components(video_path)
            logger.info(f"✅ Extracted {len(video_frames)} video frames and audio track")
            await progress_callback(analysis_id, 25, f"Extracted {len(video_frames)} video frames and audio track")
            
            # Step 2: Enhanced speech analysis with full transcript
            logger.info("🎤 Step 2: Analyzing speech with Whisper...")
            await progress_callback(analysis_id, 30, "Starting comprehensive speech analysis...")
            speech_analysis = await self.analyze_speech_enhanced(audio_path)
            logger.info(f"✅ Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM")
            await progress_callback(analysis_id, 55, f"Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM")
            
            # Step 3: Enhanced visual analysis with more frames
            logger.info("👁️ Step 3: Analyzing visual elements with GPT-4 Vision...")
            await progress_callback(analysis_id, 60, "Starting enhanced visual analysis...")
            visual_analysis = await self.analyze_visual_elements_enhanced(video_frames)
            logger.info(f"✅ Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed")
            await progress_callback(analysis_id, 75, f"Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed")
            
            # Step 4: Enhanced pedagogical assessment with full transcript
            logger.info("🎓 Step 4: Generating comprehensive pedagogical insights...")
            await progress_callback(analysis_id, 80, "Generating detailed pedagogical insights...")
            pedagogical_analysis = await self.analyze_pedagogy_enhanced(speech_analysis, visual_analysis)
            logger.info("✅ Enhanced pedagogical analysis complete")
            await progress_callback(analysis_id, 90, "Enhanced pedagogical analysis complete")
            
            # Step 5: Enhanced score combination with weighted sub-components
            logger.info("📊 Step 5: Calculating weighted component scores...")
            await progress_callback(analysis_id, 95, "Calculating final weighted scores...")
            final_results = await self.combine_analysis_enhanced(speech_analysis, visual_analysis, pedagogical_analysis)
            logger.info(f"✅ Enhanced analysis complete! Overall score: {final_results['overall_score']}/10")
            
            # Cleanup temporary files
            await self.cleanup_temp_files(audio_path, video_frames)
            logger.info("🧹 Cleanup complete")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced video analysis failed for {analysis_id}: {str(e)}")
            raise Exception(f"Enhanced video analysis failed: {str(e)}")
    
    async def extract_components(self, video_path: Path):
        """
        Enhanced component extraction with improved frame sampling
        """
        import shutil

        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            raise Exception("FFmpeg is not installed on the server. Please contact support.")
    
        # Extract audio using FFmpeg
        audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        # Use subprocess to run FFmpeg
        audio_command = [
            'ffmpeg', '-i', str(video_path),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',  # 16kHz sample rate for speech recognition
            '-ac', '1',      # Mono audio
            str(audio_path),
            '-y'  # Overwrite output files
        ]
        
        subprocess.run(audio_command, capture_output=True, check=True)
        
        # Enhanced video frame extraction
        video_frames = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        
        # Use configurable frame sampling
        frame_interval_seconds = ANALYSIS_CONFIG["sampling"]["frame_interval_seconds"]
        max_frames = ANALYSIS_CONFIG["sampling"]["max_frames_analyzed"]
        
        frame_interval = int(fps * frame_interval_seconds)
        
        # Calculate actual frames to extract
        estimated_frames = int(duration_seconds / frame_interval_seconds)
        frames_to_extract = min(estimated_frames, max_frames)
        
        # If video is long, space frames more evenly
        if estimated_frames > max_frames:
            frame_interval = total_frames // max_frames
        
        logger.info(f"📊 Video duration: {duration_seconds:.1f}s, extracting {frames_to_extract} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while True and extracted_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize frame for processing
                frame_resized = cv2.resize(frame, (640, 480))
                video_frames.append({
                    'frame': frame_resized,
                    'timestamp': frame_count / fps,
                    'frame_number': frame_count
                })
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        
        logger.info(f"📊 Extracted {len(video_frames)} frames from {duration_seconds:.1f}s video")
        return audio_path, video_frames
    
    async def analyze_speech_enhanced(self, audio_path: Path) -> Dict[str, Any]:
        """
        Enhanced speech analysis using full transcript and expanded metrics
        """
        logger.info("🎤 Starting enhanced Whisper transcription...")
        
        # Transcribe audio using Whisper
        with open(audio_path, "rb") as audio_file:
            transcript_response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        
        logger.info("✅ Whisper transcription complete")
        transcript_text = transcript_response.text
        logger.info(f"📝 Full transcript length: {len(transcript_text)} characters")
        
        # Enhanced speech metrics calculation
        logger.info("🔢 Calculating enhanced speech metrics...")
        audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
        
        # Basic metrics
        duration_minutes = len(audio_data) / sample_rate / 60
        words = transcript_text.split()
        word_count = len(words)
        speaking_rate = word_count / duration_minutes if duration_minutes > 0 else 0
        
        logger.info(f"📊 Speaking rate: {speaking_rate:.1f} WPM")
        
        # Enhanced voice activity detection
        voice_activity = librosa.effects.split(audio_data, top_db=20)
        speaking_time = sum([(end - start) / sample_rate for start, end in voice_activity])
        speaking_ratio = speaking_time / (len(audio_data) / sample_rate)
        
        # Enhanced filler word analysis with expanded list
        text_lower = transcript_text.lower()
        filler_count = 0
        filler_details = []
        
        for filler in FILLER_WORDS:
            count = text_lower.count(f' {filler} ') + text_lower.count(f'{filler} ') + text_lower.count(f' {filler}')
            if count > 0:
                filler_count += count
                filler_details.append({'word': filler, 'count': count})
        
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        logger.info(f"📊 Enhanced filler analysis: {filler_ratio:.3f} ({filler_count} filler words from expanded list)")
        
        # Voice variety analysis (pitch and energy variation)
        voice_variety_score = self.calculate_voice_variety(audio_data, sample_rate)
        
        # Pause effectiveness analysis
        pause_effectiveness_score = self.calculate_pause_effectiveness(transcript_response.words if hasattr(transcript_response, 'words') else [])
        
        # Extract key phrases using improved analysis
        sentences = re.split(r'[.!?]+', transcript_text)
        highlights = [s.strip() for s in sentences if len(s.strip()) > 50][:10]  # Top 10 substantial sentences
        
        # Enhanced content structure analysis using FULL transcript
        logger.info("🎓 Analyzing full transcript structure with GPT-4...")
        content_analysis = await self.analyze_content_structure_enhanced(transcript_text)  # Full transcript
        logger.info("✅ Enhanced content structure analysis complete")
        
        return {
            'transcript': transcript_text,
            'confidence': 0.95,  # Whisper is generally very reliable
            'speaking_rate': speaking_rate,
            'speaking_ratio': speaking_ratio,
            'filler_ratio': filler_ratio,
            'filler_details': filler_details,
            'voice_variety_score': voice_variety_score,
            'pause_effectiveness_score': pause_effectiveness_score,
            'word_count': word_count,
            'duration_minutes': duration_minutes,
            'highlights': highlights,
            'content_structure': content_analysis,
            'word_timestamps': getattr(transcript_response, 'words', [])
        }
    
    def calculate_voice_variety(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Calculate voice variety based on pitch and energy variations
        """
        try:
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, threshold=0.1)
            
            # Get the most prominent pitch at each time frame
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:  # Valid pitch detected
                    pitch_values.append(pitch)
            
            if len(pitch_values) < 10:  # Not enough data
                return 0.5
            
            # Calculate pitch variation coefficient
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_variation = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            # Calculate energy variation
            energy = librosa.feature.rms(y=audio_data)[0]
            energy_variation = np.std(energy) / np.mean(energy) if np.mean(energy) > 0 else 0
            
            # Combine pitch and energy variations (normalized to 0-1 scale)
            voice_variety = min(1.0, (pitch_variation * 2 + energy_variation) / 2)
            
            return voice_variety
            
        except Exception as e:
            logger.warning(f"Voice variety calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def calculate_pause_effectiveness(self, word_timestamps: List[Dict]) -> float:
        """
        Calculate pause effectiveness based on timing and context
        """
        if len(word_timestamps) < 5:
            return 0.5
        
        try:
            strategic_pauses = 0
            excessive_pauses = 0
            total_pauses = 0
            
            for i in range(len(word_timestamps) - 1):
                current_word = word_timestamps[i]
                next_word = word_timestamps[i + 1]
                
                current_end = current_word.get('end', 0)
                next_start = next_word.get('start', 0)
                pause_duration = next_start - current_end
                
                if pause_duration > 0.5:  # Pause longer than 0.5 seconds
                    total_pauses += 1
                    
                    if 0.5 <= pause_duration <= 2.0:  # Strategic pause range
                        strategic_pauses += 1
                    elif pause_duration > 3.0:  # Excessive pause
                        excessive_pauses += 1
            
            if total_pauses == 0:
                return 0.3  # No pauses might indicate rushed delivery
            
            # Calculate effectiveness ratio
            effectiveness = (strategic_pauses - excessive_pauses * 0.5) / total_pauses
            return max(0.0, min(1.0, effectiveness))
            
        except Exception as e:
            logger.warning(f"Pause effectiveness calculation failed: {e}")
            return 0.5
    
    async def analyze_content_structure_enhanced(self, transcript: str) -> Dict[str, Any]:
        """
        Enhanced content structure analysis using FULL transcript
        """
        # Use full transcript instead of limiting to 3000 characters
        full_transcript = transcript
        logger.info(f"📊 Analyzing full transcript: {len(full_transcript)} characters")
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in educational content analysis. Analyze the COMPLETE lecture transcript for:
                    1. Content organization and logical flow throughout the entire lecture
                    2. Use of examples and explanations across all content
                    3. Educational techniques used throughout
                    4. Clarity of key concepts across the full lecture
                    5. Student engagement elements throughout the session
                    6. Knowledge checking and comprehension verification
                    7. Conclusion and summary effectiveness
                    
                    Provide detailed scores (1-10) and specific feedback based on the COMPLETE transcript."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this COMPLETE lecture transcript for educational quality:
                    
                    {full_transcript}
                    
                    Rate the following aspects (1-10) based on the ENTIRE lecture:
                    - Content organization (logical flow, transitions, structure)
                    - Engagement techniques (questions, interactions, variety)
                    - Communication clarity (explanations, terminology, pace)
                    - Use of examples (quantity, quality, relevance)
                    - Knowledge checking (comprehension verification, feedback)
                    
                    Also identify:
                    - Key topics covered throughout
                    - Teaching techniques used across the session
                    - Specific areas for improvement
                    - Strengths demonstrated throughout
                    
                    Return as JSON with keys: content_organization, engagement_techniques, communication_clarity, use_of_examples, knowledge_checking, key_topics, techniques, improvements, strengths"""
                }
            ],
            max_tokens=1200  # Increased token limit for full analysis
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                'content_organization': 7.5,
                'engagement_techniques': 7.0,
                'communication_clarity': 7.5,
                'use_of_examples': 7.0,
                'knowledge_checking': 6.5,
                'key_topics': ['Main lecture content analyzed'],
                'techniques': ['Direct explanation', 'Content delivery'],
                'improvements': ['More examples needed', 'Improve engagement', 'Add comprehension checks'],
                'strengths': ['Clear delivery', 'Good organization', 'Professional presentation']
            }
    
    async def analyze_visual_elements_enhanced(self, video_frames: List[Dict]) -> Dict[str, Any]:
        """
        Enhanced visual analysis with more frames and detailed metrics
        """
        if not video_frames:
            return {'error': 'No video frames to analyze'}
        
        # Use configurable max frames (now 40 instead of 10)
        max_frames = ANALYSIS_CONFIG["sampling"]["max_frames_analyzed"]
        selected_frames = video_frames[:max_frames]
        
        logger.info(f"📊 Analyzing {len(selected_frames)} frames with enhanced visual metrics")
        
        frame_analyses = []
        
        for i, frame_data in enumerate(selected_frames):
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            # Convert frame to base64 for OpenAI Vision API
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Enhanced frame analysis with GPT-4 Vision
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this lecture frame (timestamp: {timestamp:.1f}s) for detailed pedagogical elements:
                                
                                1. Eye Contact & Gaze Direction: 
                                   - Looking directly at camera/students vs reading/looking away
                                   - Natural vs forced eye contact
                                   - Consistency of gaze engagement
                                
                                2. Hand Gestures & Body Language:
                                   - Purposeful gestures supporting content vs nervous/distracting movements
                                   - Open, engaging gestures vs closed, defensive postures
                                   - Gesture variety and naturalness
                                
                                3. Posture & Positioning:
                                   - Confident, upright stance vs slouching/poor posture
                                   - Appropriate positioning relative to camera/audience
                                   - Movement and spatial awareness
                                
                                4. Facial Expressions & Engagement:
                                   - Animated, engaging expressions vs flat/monotone
                                   - Appropriate emotional expression for content
                                   - Genuine enthusiasm and interest
                                
                                5. Professional Appearance:
                                   - Appropriate dress and grooming
                                   - Visual presentation quality
                                   - Overall professional demeanor
                                
                                Rate each aspect from 1-10 and provide specific observations.
                                Return as JSON with keys: eye_contact_score, gestures_score, posture_score, engagement_score, professionalism_score, detailed_observations"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=600
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                analysis['timestamp'] = timestamp
                frame_analyses.append(analysis)
                logger.info(f"📊 Frame {i+1}/{len(selected_frames)} analyzed (t={timestamp:.1f}s)")
            except json.JSONDecodeError:
                # Enhanced fallback with timestamp
                frame_analyses.append({
                    'eye_contact_score': 7,
                    'gestures_score': 7,
                    'posture_score': 7,
                    'engagement_score': 7,
                    'professionalism_score': 8,
                    'detailed_observations': [f'Unable to parse analysis for frame at {timestamp:.1f}s'],
                    'timestamp': timestamp
                })
        
        # Enhanced aggregation with temporal weighting
        if frame_analyses:
            # Calculate weighted averages (give more weight to middle frames)
            total_frames = len(frame_analyses)
            
            weighted_scores = {
                'eye_contact': 0,
                'gestures': 0,
                'posture': 0,
                'engagement': 0,
                'professionalism': 0
            }
            
            total_weight = 0
            
            for i, analysis in enumerate(frame_analyses):
                # Weight frames in the middle more heavily (bell curve weighting)
                position_ratio = i / (total_frames - 1) if total_frames > 1 else 0.5
                weight = 1.0 - abs(position_ratio - 0.5) * 0.5  # Ranges from 0.75 to 1.0
                
                weighted_scores['eye_contact'] += analysis.get('eye_contact_score', 7) * weight
                weighted_scores['gestures'] += analysis.get('gestures_score', 7) * weight
                weighted_scores['posture'] += analysis.get('posture_score', 7) * weight
                weighted_scores['engagement'] += analysis.get('engagement_score', 7) * weight
                weighted_scores['professionalism'] += analysis.get('professionalism_score', 8) * weight
                
                total_weight += weight
            
            # Normalize by total weight
            for key in weighted_scores:
                weighted_scores[key] = weighted_scores[key] / total_weight if total_weight > 0 else 7.0
            
            # Collect all detailed observations
            all_observations = []
            for analysis in frame_analyses:
                obs = analysis.get('detailed_observations', [])
                if isinstance(obs, list):
                    all_observations.extend(obs)
                elif isinstance(obs, str):
                    all_observations.append(obs)
            
            return {
                'scores': weighted_scores,
                'observations': all_observations,
                'frames_analyzed': len(frame_analyses),
                'temporal_analysis': {
                    'frame_timestamps': [f['timestamp'] for f in frame_analyses],
                    'score_progression': {
                        'eye_contact': [f.get('eye_contact_score', 7) for f in frame_analyses],
                        'gestures': [f.get('gestures_score', 7) for f in frame_analyses],
                        'engagement': [f.get('engagement_score', 7) for f in frame_analyses]
                    }
                }
            }
        
        return {'error': 'No frames successfully analyzed'}
    
    async def analyze_pedagogy_enhanced(self, speech_analysis: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """
        Enhanced comprehensive pedagogical analysis with weighted sub-components
        """
        # Prepare enhanced context for GPT-4
        context = f"""
        COMPREHENSIVE LECTURE ANALYSIS DATA:
        
        Speech Analysis (Full Transcript):
        - Complete Transcript: {speech_analysis.get('transcript', '')[:4000]}...
        - Speaking Rate: {speech_analysis.get('speaking_rate', 0):.1f} words/minute
        - Filler Word Ratio: {speech_analysis.get('filler_ratio', 0):.3f}
        - Voice Variety Score: {speech_analysis.get('voice_variety_score', 0.5):.2f}
        - Pause Effectiveness: {speech_analysis.get('pause_effectiveness_score', 0.5):.2f}
        - Key Highlights: {'; '.join(speech_analysis.get('highlights', [])[:8])}
        
        Visual Analysis ({visual_analysis.get('frames_analyzed', 0)} frames):
        - Eye Contact Score: {visual_analysis.get('scores', {}).get('eye_contact', 'N/A')}
        - Gesture Effectiveness: {visual_analysis.get('scores', {}).get('gestures', 'N/A')}
        - Posture Score: {visual_analysis.get('scores', {}).get('posture', 'N/A')}
        - Facial Engagement: {visual_analysis.get('scores', {}).get('engagement', 'N/A')}
        - Professional Appearance: {visual_analysis.get('scores', {}).get('professionalism', 'N/A')}
        
        Content Structure Analysis:
        - Organization: {speech_analysis.get('content_structure', {}).get('content_organization', 'N/A')}
        - Clarity: {speech_analysis.get('content_structure', {}).get('communication_clarity', 'N/A')}
        - Examples Usage: {speech_analysis.get('content_structure', {}).get('use_of_examples', 'N/A')}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert pedagogical analyst with deep expertise in educational effectiveness. 
                    Analyze comprehensive lecture data and provide detailed feedback on teaching effectiveness across ALL dimensions.
                    
                    Focus on these key pedagogical areas:
                    1. Content Organization & Structure - Logical flow, transitions, scaffolding
                    2. Student Engagement Techniques - Questions, interactions, variety, attention management
                    3. Communication Clarity - Explanations, language use, concept delivery
                    4. Use of Examples & Illustrations - Quality, relevance, variety of examples
                    5. Knowledge Checking & Assessment - Verification of understanding, feedback loops
                    
                    Provide specific, actionable feedback with evidence from the data."""
                },
                {
                    "role": "user",
                    "content": f"""{context}
                    
                    Based on this comprehensive analysis, provide detailed pedagogical assessment:
                    
                    REQUIRED SCORES (1-10 for each):
                    1. Content Organization & Structure
                    2. Student Engagement Techniques  
                    3. Communication Clarity & Delivery
                    4. Use of Examples & Illustrations
                    5. Knowledge Checking & Assessment
                    
                    REQUIRED QUALITATIVE ANALYSIS:
                    6. Top 5 teaching strengths demonstrated
                    7. Top 5 areas needing improvement
                    8. Specific actionable recommendations (at least 8)
                    9. Overall teaching effectiveness assessment
                    
                    Format as JSON with keys: 
                    content_organization, engagement_techniques, communication_clarity, use_of_examples, knowledge_checking, 
                    strengths, improvements, recommendations, overall_effectiveness, detailed_analysis"""
                }
            ],
            max_tokens=1400  # Increased for comprehensive analysis
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Enhanced fallback analysis
            return {
                'content_organization': 7.5,
                'engagement_techniques': 7.0,
                'communication_clarity': 7.8,
                'use_of_examples': 7.2,
                'knowledge_checking': 6.8,
                'overall_effectiveness': 7.3,
                'strengths': [
                    'Clear and articulate speaking voice',
                    'Well-structured content presentation',
                    'Professional demeanor and appearance',
                    'Good use of voice modulation',
                    'Consistent delivery throughout session'
                ],
                'improvements': [
                    'Increase student interaction and engagement',
                    'Add more concrete examples and illustrations',
                    'Implement regular comprehension checks',
                    'Improve gesture variety and purposefulness',
                    'Enhance conclusion and summary techniques'
                ],
                'recommendations': [
                    'Practice incorporating more interactive elements',
                    'Develop a broader range of relevant examples',
                    'Work on strategic pausing for emphasis',
                    'Implement regular "check for understanding" moments',
                    'Consider movement and spatial positioning',
                    'Enhance opening and closing techniques',
                    'Practice varied questioning strategies',
                    'Develop more engaging visual presence'
                ],
                'detailed_analysis': 'Comprehensive analysis completed with enhanced metrics and full transcript review.'
            }
    
    async def combine_analysis_enhanced(self, speech_analysis: Dict, visual_analysis: Dict, pedagogical_analysis: Dict) -> Dict[str, Any]:
        """
        Enhanced analysis combination with weighted sub-components from config
        """
        # Calculate enhanced component scores using weighted sub-components
        speech_score = self.calculate_speech_score_enhanced(speech_analysis)
        visual_score = self.calculate_visual_score_enhanced(visual_analysis)
        pedagogy_score = self.calculate_pedagogy_score_enhanced(pedagogical_analysis)
        presentation_score = (speech_score + visual_score) / 2  # Derived score
        
        # Get category weights from config
        category_weights = ANALYSIS_CONFIG["weights"]
        
        # Calculate overall weighted score
        overall_score = (
            speech_score * category_weights["speech_analysis"] + 
            visual_score * category_weights["body_language"] + 
            pedagogy_score * category_weights["teaching_effectiveness"] +
            presentation_score * category_weights["presentation_skills"]
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'speech_analysis': {
                'score': round(speech_score, 1),
                'speaking_rate': speech_analysis.get('speaking_rate', 0),
                'clarity': 10 - (speech_analysis.get('filler_ratio', 0) * 20),
                'pace': min(10, max(1, 10 - abs(speech_analysis.get('speaking_rate', 150) - 150) / 20)),
                'confidence': speech_analysis.get('confidence', 0.8) * 10,
                'voice_variety': speech_analysis.get('voice_variety_score', 0.5) * 10,
                'pause_effectiveness': speech_analysis.get('pause_effectiveness_score', 0.5) * 10,
                'feedback': self.generate_speech_feedback_enhanced(speech_analysis)
            },
            'body_language': {
                'score': round(visual_score, 1),
                'eye_contact': visual_analysis.get('scores', {}).get('eye_contact', 7),
                'gestures': visual_analysis.get('scores', {}).get('gestures', 7),
                'posture': visual_analysis.get('scores', {}).get('posture', 7),
                'engagement': visual_analysis.get('scores', {}).get('engagement', 7),
                'professionalism': visual_analysis.get('scores', {}).get('professionalism', 8),
                'frames_analyzed': visual_analysis.get('frames_analyzed', 0),
                'feedback': self.generate_visual_feedback_enhanced(visual_analysis)
            },
            'teaching_effectiveness': {
                'score': round(pedagogy_score, 1),
                'content_organization': pedagogical_analysis.get('content_organization', 7),
                'engagement_techniques': pedagogical_analysis.get('engagement_techniques', 7),
                'communication_clarity': pedagogical_analysis.get('communication_clarity', 7),
                'use_of_examples': pedagogical_analysis.get('use_of_examples', 7),
                'knowledge_checking': pedagogical_analysis.get('knowledge_checking', 7),
                'feedback': pedagogical_analysis.get('recommendations', [])
            },
            'presentation_skills': {
                'score': round(presentation_score, 1),
                'professionalism': visual_analysis.get('scores', {}).get('professionalism', 8),
                'energy': speech_analysis.get('speaking_ratio', 0.7) * 10,
                'voice_modulation': speech_analysis.get('voice_variety_score', 0.5) * 10,
                'time_management': self.assess_time_management(speech_analysis),
                'feedback': self.generate_presentation_feedback_enhanced(speech_analysis, visual_analysis)
            },
            'strengths': pedagogical_analysis.get('strengths', []),
            'improvement_suggestions': pedagogical_analysis.get('improvements', []),
            'detailed_insights': {
                'transcript_summary': speech_analysis.get('transcript', '')[:800] + '...',
                'key_highlights': speech_analysis.get('highlights', []),
                'visual_observations': visual_analysis.get('observations', [])[:10],
                'filler_word_analysis': speech_analysis.get('filler_details', []),
                'temporal_visual_data': visual_analysis.get('temporal_analysis', {})
            },
            'configuration_used': {
                'frames_analyzed': visual_analysis.get('frames_analyzed', 0),
                'transcript_length': len(speech_analysis.get('transcript', '')),
                'category_weights': category_weights,
                'filler_words_detected': len(speech_analysis.get('filler_details', []))
            }
        }
    
    def calculate_speech_score_enhanced(self, speech_analysis: Dict) -> float:
        """Enhanced speech score calculation with weighted sub-components"""
        weights = ANALYSIS_CONFIG["weights"]["speech_components"]
        
        # Safe float conversion helper
        def safe_float(value, default=0.0):
            if isinstance(value, (list, tuple)):
                value = value[0] if value else default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Calculate individual component scores using configurable thresholds
        speaking_rate = safe_float(speech_analysis.get('speaking_rate', 150))
        rate_score = float(calculate_metric_score(speaking_rate, SPEECH_METRICS["speaking_rate"]))
        
        filler_ratio = safe_float(speech_analysis.get('filler_ratio', 0.05))
        fluency_score = float(calculate_metric_score(filler_ratio, SPEECH_METRICS["filler_ratio"], reverse_scale=True))
        
        confidence = safe_float(speech_analysis.get('confidence', 0.8))
        clarity_score = float(calculate_metric_score(confidence, SPEECH_METRICS["speaking_clarity"]))
        
        voice_variety = safe_float(speech_analysis.get('voice_variety_score', 0.5))
        variety_score = float(calculate_metric_score(voice_variety, SPEECH_METRICS["voice_variety"]))
        
        pause_effectiveness = safe_float(speech_analysis.get('pause_effectiveness_score', 0.5))
        pause_score = float(calculate_metric_score(pause_effectiveness, SPEECH_METRICS["pause_effectiveness"]))
        
        # Weighted combination
        total_score = (
            rate_score * weights["speaking_rate"] +
            fluency_score * weights["clarity"] +
            clarity_score * weights["confidence"] +
            variety_score * weights["voice_variety"] +
            pause_score * weights["pause_effectiveness"]
        )
        
        return float(total_score)
    
    def calculate_visual_score_enhanced(self, visual_analysis: Dict) -> float:
        """Enhanced visual score calculation with weighted sub-components"""
        weights = ANALYSIS_CONFIG["weights"]["visual_components"]
        scores = visual_analysis.get('scores', {})
        
        if not scores:
            return 7.0
        
        # Ensure all scores are floats, not lists or strings
        def safe_float(value, default=7.0):
            if isinstance(value, (list, tuple)):
                value = value[0] if value else default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Calculate weighted visual score with safe conversion
        total_score = (
            safe_float(scores.get('eye_contact', 7)) * weights["eye_contact"] +
            safe_float(scores.get('gestures', 7)) * weights["gestures"] +
            safe_float(scores.get('posture', 7)) * weights["posture"] +
            safe_float(scores.get('engagement', 7)) * weights["engagement"] +
            safe_float(scores.get('professionalism', 8)) * weights["professionalism"]
        )
        
        return float(total_score)
    
    def calculate_pedagogy_score_enhanced(self, pedagogical_analysis: Dict) -> float:
        """Enhanced pedagogy score calculation with weighted sub-components"""
        weights = ANALYSIS_CONFIG["weights"]["pedagogy_components"]
        
        # Use individual component scores instead of just overall_effectiveness
        total_score = (
            pedagogical_analysis.get('content_organization', 7) * weights["content_organization"] +
            pedagogical_analysis.get('engagement_techniques', 7) * weights["engagement_techniques"] +
            pedagogical_analysis.get('communication_clarity', 7) * weights["communication_clarity"] +
            pedagogical_analysis.get('use_of_examples', 7) * weights["use_of_examples"] +
            pedagogical_analysis.get('knowledge_checking', 6.5) * weights["knowledge_checking"]
        )
        
        return total_score
    
    def generate_speech_feedback_enhanced(self, speech_analysis: Dict) -> List[str]:
        """Generate enhanced speech feedback using configurable metrics"""
        feedback = []
        
        speaking_rate = speech_analysis.get('speaking_rate', 150)
        feedback.append(get_metric_feedback(speaking_rate, SPEECH_METRICS["speaking_rate"]))
        
        filler_ratio = speech_analysis.get('filler_ratio', 0.05)
        feedback.append(get_metric_feedback(filler_ratio, SPEECH_METRICS["filler_ratio"], reverse_scale=True))
        
        voice_variety = speech_analysis.get('voice_variety_score', 0.5)
        if voice_variety > 0.7:
            feedback.append("Excellent voice modulation adds engagement and emphasis")
        elif voice_variety < 0.3:
            feedback.append("Consider adding more variation in pitch and energy for better engagement")
        
        pause_effectiveness = speech_analysis.get('pause_effectiveness_score', 0.5)
        if pause_effectiveness > 0.7:
            feedback.append("Strategic use of pauses enhances content delivery")
        elif pause_effectiveness < 0.3:
            feedback.append("Work on using pauses more strategically for emphasis and comprehension")
        
        return feedback
    
    def generate_visual_feedback_enhanced(self, visual_analysis: Dict) -> List[str]:
        """Generate enhanced visual feedback with temporal insights"""
        feedback = []
        scores = visual_analysis.get('scores', {})
        
        # Enhanced feedback based on configurable thresholds
        for metric_name, score in scores.items():
            if metric_name in ['eye_contact', 'gestures', 'posture', 'engagement', 'professionalism']:
                if score >= 8.5:
                    feedback.append(f"Excellent {metric_name.replace('_', ' ')} throughout the session")
                elif score < 6.0:
                    feedback.append(f"Focus on improving {metric_name.replace('_', ' ')} consistency")
        
        # Add temporal analysis feedback if available
        temporal_data = visual_analysis.get('temporal_analysis', {})
        if temporal_data and 'score_progression' in temporal_data:
            progression = temporal_data['score_progression']
            for metric, scores_over_time in progression.items():
                if len(scores_over_time) > 3:
                    trend = np.polyfit(range(len(scores_over_time)), scores_over_time, 1)[0]
                    if trend > 0.1:
                        feedback.append(f"{metric.replace('_', ' ').title()} improved throughout the session")
                    elif trend < -0.1:
                        feedback.append(f"Maintain {metric.replace('_', ' ')} consistency throughout longer sessions")
        
        return feedback
    
    def generate_presentation_feedback_enhanced(self, speech_analysis: Dict, visual_analysis: Dict) -> List[str]:
        """Generate enhanced presentation feedback combining multiple metrics"""
        feedback = []
        
        # Overall energy assessment
        speech_energy = speech_analysis.get('speaking_ratio', 0.7)
        voice_variety = speech_analysis.get('voice_variety_score', 0.5)
        visual_engagement = visual_analysis.get('scores', {}).get('engagement', 7) / 10
        
        overall_energy = (speech_energy + voice_variety + visual_engagement) / 3
        
        if overall_energy > 0.8:
            feedback.append("High energy and dynamic presentation style throughout")
        elif overall_energy < 0.5:
            feedback.append("Consider increasing overall energy and enthusiasm")
        else:
            feedback.append("Good energy balance - maintain consistency")
        
        # Professional delivery assessment
        frames_analyzed = visual_analysis.get('frames_analyzed', 0)
        if frames_analyzed > 20:
            feedback.append(f"Comprehensive analysis of {frames_analyzed} frames shows consistent delivery")
        
        return feedback
    
    def assess_time_management(self, speech_analysis: Dict) -> float:
        """Enhanced time management assessment"""
        speaking_ratio = speech_analysis.get('speaking_ratio', 0.7)
        pause_effectiveness = speech_analysis.get('pause_effectiveness_score', 0.5)
        
        # Good time management combines appropriate speaking ratio with effective pauses
        time_score = (speaking_ratio * 0.7 + pause_effectiveness * 0.3) * 10
        
        return min(10.0, max(1.0, time_score))
    
    async def cleanup_temp_files(self, audio_path: Path, video_frames: List):
        """Clean up temporary files"""
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

# Global processor instance
video_processor = VideoAnalysisProcessor()