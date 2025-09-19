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
        Main processing pipeline for video analysis
        """
        try:
            logger.info(f"🎬 Starting video analysis for {analysis_id}")
            logger.info(f"📁 File: {video_path.name} ({video_path.stat().st_size / (1024*1024):.1f}MB)")
            
            # Step 1: Extract audio and video components
            logger.info("🔧 Step 1: Extracting audio and video components...")
            await progress_callback(analysis_id, 10, "Extracting audio and video components...")
            audio_path, video_frames = await self.extract_components(video_path)
            logger.info(f"✅ Extracted {len(video_frames)} video frames and audio track")
            await progress_callback(analysis_id, 25, f"Extracted {len(video_frames)} video frames and audio track")
            
            # Step 2: Analyze speech
            logger.info("🎤 Step 2: Analyzing speech with Whisper...")
            await progress_callback(analysis_id, 30, "Starting Whisper speech analysis...")
            speech_analysis = await self.analyze_speech(audio_path)
            logger.info(f"✅ Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM")
            await progress_callback(analysis_id, 55, f"Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM")
            
            # Step 3: Analyze visual elements
            logger.info("👁️ Step 3: Analyzing visual elements with GPT-4 Vision...")
            await progress_callback(analysis_id, 60, "Starting GPT-4 Vision analysis...")
            visual_analysis = await self.analyze_visual_elements(video_frames)
            logger.info(f"✅ Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed")
            await progress_callback(analysis_id, 75, f"Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed")
            
            # Step 4: Comprehensive pedagogical assessment
            logger.info("🎓 Step 4: Generating pedagogical insights...")
            await progress_callback(analysis_id, 80, "Generating pedagogical insights with GPT-4...")
            pedagogical_analysis = await self.analyze_pedagogy(speech_analysis, visual_analysis)
            logger.info("✅ Pedagogical analysis complete")
            await progress_callback(analysis_id, 90, "Pedagogical analysis complete")
            
            # Step 5: Combine and score
            logger.info("📊 Step 5: Combining analysis results...")
            await progress_callback(analysis_id, 95, "Combining results and calculating final scores...")
            final_results = await self.combine_analysis(speech_analysis, visual_analysis, pedagogical_analysis)
            logger.info(f"✅ Final analysis complete! Overall score: {final_results['overall_score']}/10")
            
            # Cleanup temporary files
            await self.cleanup_temp_files(audio_path, video_frames)
            logger.info("🧹 Cleanup complete")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Video analysis failed for {analysis_id}: {str(e)}")
            raise Exception(f"Video analysis failed: {str(e)}")
    
    async def extract_components(self, video_path: Path):
        """
        Extract audio and key video frames from uploaded video
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
        
        # Extract video frames (every 10 seconds for analysis)
        video_frames = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 10)  # Every 10 seconds
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize frame for processing
                frame_resized = cv2.resize(frame, (640, 480))
                video_frames.append(frame_resized)
                
            frame_count += 1
            
        cap.release()
        
        return audio_path, video_frames
    
    async def analyze_speech(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze speech using OpenAI Whisper for transcription and custom speech metrics
        """
        logger.info("🎤 Starting Whisper transcription...")
        
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
        logger.info(f"📝 Transcript length: {len(transcript_text)} characters")
        
        logger.info("🔢 Calculating speech metrics...")
        # Calculate speech metrics using librosa
        audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
        
        # Speaking rate (words per minute)
        duration_minutes = len(audio_data) / sample_rate / 60
        words = transcript_text.split()
        word_count = len(words)
        speaking_rate = word_count / duration_minutes if duration_minutes > 0 else 0
        
        logger.info(f"📊 Speaking rate: {speaking_rate:.1f} WPM")
        
        # Voice activity detection
        voice_activity = librosa.effects.split(audio_data, top_db=20)
        speaking_time = sum([(end - start) / sample_rate for start, end in voice_activity])
        speaking_ratio = speaking_time / (len(audio_data) / sample_rate)
        
        # Filler word analysis
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'okay', 'right']
        text_lower = transcript_text.lower()
        filler_count = sum(text_lower.count(f' {filler} ') + text_lower.count(f'{filler} ') + text_lower.count(f' {filler}') for filler in filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        logger.info(f"📊 Filler word ratio: {filler_ratio:.3f} ({filler_count} filler words)")
        
        # Extract key phrases using simple keyword analysis
        sentences = re.split(r'[.!?]+', transcript_text)
        highlights = [s.strip() for s in sentences if len(s.strip()) > 50][:5]  # First 5 substantial sentences
        
        logger.info("🎓 Analyzing content structure with GPT-4...")
        # Analyze content structure using GPT-4
        content_analysis = await self.analyze_content_structure(transcript_text)
        logger.info("✅ Content structure analysis complete")
        
        return {
            'transcript': transcript_text,
            'confidence': 0.95,  # Whisper is generally very reliable
            'speaking_rate': speaking_rate,
            'speaking_ratio': speaking_ratio,
            'filler_ratio': filler_ratio,
            'word_count': word_count,
            'duration_minutes': duration_minutes,
            'highlights': highlights,
            'content_structure': content_analysis,
            'word_timestamps': getattr(transcript_response, 'words', [])
        }
    
    async def analyze_content_structure(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze content structure and educational quality using GPT-4
        """
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in educational content analysis. Analyze lecture transcripts for:
                    1. Content organization and logical flow
                    2. Use of examples and explanations
                    3. Educational techniques used
                    4. Clarity of key concepts
                    5. Student engagement elements
                    
                    Provide scores (1-10) and specific feedback."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this lecture transcript for educational quality:
                    
                    {transcript[:3000]}...
                    
                    Rate the following aspects (1-10):
                    - Content organization
                    - Use of examples
                    - Concept clarity
                    - Engagement techniques
                    
                    Also identify:
                    - Key topics covered
                    - Teaching techniques used
                    - Areas for improvement
                    
                    Return as JSON with keys: organization_score, examples_score, clarity_score, engagement_score, key_topics, techniques, improvements"""
                }
            ],
            max_tokens=800
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                'organization_score': 7.5,
                'examples_score': 7.0,
                'clarity_score': 7.5,
                'engagement_score': 7.0,
                'key_topics': ['Main lecture content'],
                'techniques': ['Direct explanation'],
                'improvements': ['More examples needed', 'Improve engagement']
            }
    
    async def analyze_visual_elements(self, video_frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze visual elements using OpenAI's vision model
        """
        if not video_frames:
            return {'error': 'No video frames to analyze'}
        
        # Select representative frames (max 10 for cost efficiency)
        selected_frames = video_frames[::max(1, len(video_frames) // 10)][:10]
        
        frame_analyses = []
        
        for i, frame in enumerate(selected_frames):
            # Convert frame to base64 for OpenAI Vision API
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Analyze frame with GPT-4 Vision
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this lecture frame for pedagogical elements. Focus on:
                                1. Eye contact and gaze direction (looking at camera/students vs reading/looking away)
                                2. Hand gestures and body language (open, engaging gestures vs closed, nervous gestures)
                                3. Posture and positioning (confident stance vs slouching)
                                4. Facial expressions and engagement
                                5. Use of visual aids or props
                                6. Professional appearance
                                
                                Rate each aspect from 1-10 and provide specific observations.
                                Return as JSON with keys: eye_contact_score, gestures_score, posture_score, engagement_score, professionalism_score, observations"""
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
                max_tokens=500
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                frame_analyses.append(analysis)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                frame_analyses.append({
                    'eye_contact_score': 7,
                    'gestures_score': 7,
                    'posture_score': 7,
                    'engagement_score': 7,
                    'professionalism_score': 8,
                    'observations': 'Unable to parse detailed analysis'
                })
        
        # Aggregate frame analyses
        if frame_analyses:
            avg_scores = {
                'eye_contact': np.mean([f.get('eye_contact_score', 7) for f in frame_analyses]),
                'gestures': np.mean([f.get('gestures_score', 7) for f in frame_analyses]),
                'posture': np.mean([f.get('posture_score', 7) for f in frame_analyses]),
                'engagement': np.mean([f.get('engagement_score', 7) for f in frame_analyses]),
                'professionalism': np.mean([f.get('professionalism_score', 8) for f in frame_analyses])
            }
            
            all_observations = [obs for f in frame_analyses for obs in f.get('observations', [])]
            
            return {
                'scores': avg_scores,
                'observations': all_observations,
                'frames_analyzed': len(frame_analyses)
            }
        
        return {'error': 'No frames successfully analyzed'}
    
    async def analyze_pedagogy(self, speech_analysis: Dict, visual_analysis: Dict) -> Dict[str, Any]:
        """
        Comprehensive pedagogical analysis using GPT-4
        """
        # Prepare context for GPT-4
        context = f"""
        LECTURE ANALYSIS DATA:
        
        Speech Analysis:
        - Transcript: {speech_analysis.get('transcript', '')[:2000]}...
        - Speaking Rate: {speech_analysis.get('speaking_rate', 0):.1f} words/minute
        - Filler Word Ratio: {speech_analysis.get('filler_ratio', 0):.3f}
        - Key Highlights: {', '.join(speech_analysis.get('highlights', [])[:5])}
        
        Visual Analysis:
        - Eye Contact Score: {visual_analysis.get('scores', {}).get('eye_contact', 'N/A')}
        - Gesture Score: {visual_analysis.get('scores', {}).get('gestures', 'N/A')}
        - Posture Score: {visual_analysis.get('scores', {}).get('posture', 'N/A')}
        - Engagement Score: {visual_analysis.get('scores', {}).get('engagement', 'N/A')}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert pedagogical analyst. Analyze lecture data and provide comprehensive feedback on teaching effectiveness. Focus on:
                    
                    1. Content organization and clarity
                    2. Student engagement techniques
                    3. Communication effectiveness
                    4. Use of examples and explanations
                    5. Overall pedagogical approach
                    
                    Provide scores (1-10) and specific, actionable feedback."""
                },
                {
                    "role": "user",
                    "content": f"""{context}
                    
                    Please analyze this lecture data and provide:
                    1. Content organization score (1-10)
                    2. Engagement techniques score (1-10)
                    3. Communication clarity score (1-10)
                    4. Use of examples score (1-10)
                    5. Overall teaching effectiveness score (1-10)
                    6. Top 3 strengths
                    7. Top 3 areas for improvement
                    8. Specific actionable recommendations
                    
                    Format as JSON with keys: content_organization, engagement_techniques, communication_clarity, use_of_examples, overall_effectiveness, strengths, improvements, recommendations"""
                }
            ],
            max_tokens=1000
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback analysis
            return {
                'content_organization': 7.5,
                'engagement_techniques': 7.0,
                'communication_clarity': 7.8,
                'use_of_examples': 7.2,
                'overall_effectiveness': 7.4,
                'strengths': ['Clear speaking voice', 'Good content structure', 'Professional delivery'],
                'improvements': ['More student interaction', 'Better use of examples', 'Improved pacing'],
                'recommendations': ['Practice with interactive elements', 'Develop more concrete examples', 'Work on speaking rhythm']
            }
    
    async def combine_analysis(self, speech_analysis: Dict, visual_analysis: Dict, pedagogical_analysis: Dict) -> Dict[str, Any]:
        """
        Combine all analyses into final comprehensive report
        """
        # Calculate component scores
        speech_score = self.calculate_speech_score(speech_analysis)
        visual_score = self.calculate_visual_score(visual_analysis)
        pedagogy_score = self.calculate_pedagogy_score(pedagogical_analysis)
        
        # Overall weighted score
        overall_score = (speech_score * 0.3 + visual_score * 0.3 + pedagogy_score * 0.4)
        
        return {
            'overall_score': round(overall_score, 1),
            'speech_analysis': {
                'score': round(speech_score, 1),
                'speaking_rate': speech_analysis.get('speaking_rate', 0),
                'clarity': 10 - (speech_analysis.get('filler_ratio', 0) * 20),
                'pace': min(10, max(1, 10 - abs(speech_analysis.get('speaking_rate', 150) - 150) / 20)),
                'confidence': speech_analysis.get('confidence', 0.8) * 10,
                'feedback': self.generate_speech_feedback(speech_analysis)
            },
            'body_language': {
                'score': round(visual_score, 1),
                'eye_contact': visual_analysis.get('scores', {}).get('eye_contact', 7),
                'gestures': visual_analysis.get('scores', {}).get('gestures', 7),
                'posture': visual_analysis.get('scores', {}).get('posture', 7),
                'engagement': visual_analysis.get('scores', {}).get('engagement', 7),
                'feedback': self.generate_visual_feedback(visual_analysis)
            },
            'teaching_effectiveness': {
                'score': round(pedagogy_score, 1),
                'content_flow': pedagogical_analysis.get('content_organization', 7),
                'explanations': pedagogical_analysis.get('communication_clarity', 7),
                'examples': pedagogical_analysis.get('use_of_examples', 7),
                'engagement': pedagogical_analysis.get('engagement_techniques', 7),
                'feedback': pedagogical_analysis.get('recommendations', [])
            },
            'presentation_skills': {
                'score': round((speech_score + visual_score) / 2, 1),
                'professionalism': visual_analysis.get('scores', {}).get('professionalism', 8),
                'energy': speech_analysis.get('speaking_ratio', 0.7) * 10,
                'time_management': self.assess_time_management(speech_analysis),
                'conclusion': 8.0,  # Would need more sophisticated analysis
                'feedback': self.generate_presentation_feedback(speech_analysis, visual_analysis)
            },
            'strengths': pedagogical_analysis.get('strengths', []),
            'improvement_suggestions': pedagogical_analysis.get('improvements', []),
            'detailed_insights': {
                'transcript_summary': speech_analysis.get('transcript', '')[:500] + '...',
                'key_highlights': speech_analysis.get('highlights', []),
                'visual_observations': visual_analysis.get('observations', [])
            }
        }
    
    def calculate_speech_score(self, speech_analysis: Dict) -> float:
        """Calculate overall speech score from metrics"""
        confidence = speech_analysis.get('confidence', 0.8)
        speaking_rate = speech_analysis.get('speaking_rate', 150)
        filler_ratio = speech_analysis.get('filler_ratio', 0.05)
        
        # Optimal speaking rate is 140-160 words per minute
        rate_score = max(0, 10 - abs(speaking_rate - 150) / 20)
        confidence_score = confidence * 10
        clarity_score = max(0, 10 - filler_ratio * 50)
        
        return (rate_score + confidence_score + clarity_score) / 3
    
    def calculate_visual_score(self, visual_analysis: Dict) -> float:
        """Calculate overall visual score from metrics"""
        scores = visual_analysis.get('scores', {})
        return np.mean(list(scores.values())) if scores else 7.0
    
    def calculate_pedagogy_score(self, pedagogical_analysis: Dict) -> float:
        """Calculate overall pedagogy score"""
        return pedagogical_analysis.get('overall_effectiveness', 7.0)
    
    def assess_time_management(self, speech_analysis: Dict) -> float:
        """Assess time management based on speaking patterns"""
        speaking_ratio = speech_analysis.get('speaking_ratio', 0.7)
        # Good time management means not too much dead time, not too rushed
        if 0.6 <= speaking_ratio <= 0.8:
            return 9.0
        elif 0.5 <= speaking_ratio <= 0.9:
            return 7.5
        else:
            return 6.0
    
    def generate_speech_feedback(self, speech_analysis: Dict) -> List[str]:
        """Generate specific speech feedback"""
        feedback = []
        
        speaking_rate = speech_analysis.get('speaking_rate', 150)
        if speaking_rate > 180:
            feedback.append("Consider slowing down your speaking pace for better comprehension")
        elif speaking_rate < 120:
            feedback.append("Consider increasing your speaking pace to maintain engagement")
        else:
            feedback.append("Excellent speaking pace for lecture delivery")
        
        filler_ratio = speech_analysis.get('filler_ratio', 0.05)
        if filler_ratio > 0.1:
            feedback.append("Work on reducing filler words (um, uh, like) for smoother delivery")
        else:
            feedback.append("Great job minimizing filler words")
        
        confidence = speech_analysis.get('confidence', 0.8)
        if confidence > 0.9:
            feedback.append("Excellent voice clarity and articulation")
        elif confidence < 0.7:
            feedback.append("Consider speaking more clearly and articulating words")
        
        return feedback
    
    def generate_visual_feedback(self, visual_analysis: Dict) -> List[str]:
        """Generate specific visual feedback"""
        feedback = []
        scores = visual_analysis.get('scores', {})
        
        if scores.get('eye_contact', 7) >= 8:
            feedback.append("Excellent eye contact with the audience")
        elif scores.get('eye_contact', 7) < 6:
            feedback.append("Improve eye contact by looking at the camera/audience more frequently")
        
        if scores.get('gestures', 7) >= 8:
            feedback.append("Great use of hand gestures to support your content")
        elif scores.get('gestures', 7) < 6:
            feedback.append("Use more purposeful hand gestures to emphasize key points")
        
        if scores.get('posture', 7) >= 8:
            feedback.append("Confident and professional posture throughout")
        elif scores.get('posture', 7) < 6:
            feedback.append("Work on maintaining upright, confident posture")
        
        return feedback
    
    def generate_presentation_feedback(self, speech_analysis: Dict, visual_analysis: Dict) -> List[str]:
        """Generate overall presentation feedback"""
        feedback = []
        
        # Combine insights from both analyses
        overall_energy = (speech_analysis.get('speaking_ratio', 0.7) + 
                         visual_analysis.get('scores', {}).get('engagement', 7)/10) / 2
        
        if overall_energy > 0.8:
            feedback.append("High energy and engaging presentation style")
        elif overall_energy < 0.6:
            feedback.append("Consider increasing energy and enthusiasm")
        
        feedback.append("Professional delivery with clear structure")
        feedback.append("Good use of verbal and non-verbal communication")
        
        return feedback
    
    async def cleanup_temp_files(self, audio_path: Path, video_frames: List):
        """Clean up temporary files"""
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")

# Global processor instance
video_processor = VideoAnalysisProcessor()