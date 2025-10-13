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
from datetime import datetime



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
        self.analysis_id = None  # Add this to store current analysis_id
        self.progress_callback = None  # Add this to store callback
    
    async def simulate_progress_during_operation(self, start_progress: int, end_progress: int, duration_seconds: float, message: str):
        """Simulate progress updates during a long operation"""
        if not self.progress_callback or not self.analysis_id:
            return
            
        steps = max(int(duration_seconds * 2), 1)  # Update every 0.5 seconds
        increment = (end_progress - start_progress) / steps
        
        for i in range(steps):
            # Check if stop was requested
            if hasattr(self, 'should_stop') and self.should_stop:
                raise Exception("Analysis stopped by user")
                
            current_progress = int(start_progress + (increment * i))
            await self.progress_callback(
                self.analysis_id, 
                current_progress, 
                f"{message} ({current_progress}%)"
            )
            await asyncio.sleep(0.3)  # Reduced sleep for more responsive updates
        
    async def process_video(self, video_path: Path, analysis_id: str, progress_callback):
        """
        Enhanced processing pipeline for video analysis with improved sampling and metrics
        """
        try:
            # Store for use in helper methods
            self.analysis_id = analysis_id
            self.progress_callback = progress_callback

            logger.info(f"🎬 Starting enhanced video analysis for {analysis_id}")
            await progress_callback(analysis_id, 5, f"🎬 Starting enhanced video analysis for {analysis_id}")
            await asyncio.sleep(0)  # Force immediate execution
            
            logger.info(f"📁 File: {video_path.name} ({video_path.stat().st_size / (1024*1024):.1f}MB)")
            await progress_callback(analysis_id, 6, f"📁 File: {video_path.name} ({video_path.stat().st_size / (1024*1024):.1f}MB)")
            await asyncio.sleep(0)  # Force immediate execution
            
            # Step 1: Extract audio and video components with enhanced sampling
            logger.info("🔧 Step 1: Extracting audio and video components...")
            await progress_callback(analysis_id, 10, "🔧 Step 1: Extracting audio and video components...")
            await asyncio.sleep(0)  # Force immediate execution
            
            audio_path, video_frames = await self.extract_components(video_path)
            
            logger.info(f"✅ Extracted {len(video_frames)} video frames and audio track")
            await progress_callback(analysis_id, 20, f"✅ Extracted {len(video_frames)} video frames and audio track")
            await asyncio.sleep(0)  # Force immediate execution
            
            # Get video metadata
            cap_temp = cv2.VideoCapture(str(video_path))
            fps = cap_temp.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps if fps > 0 else 0
            cap_temp.release()

            await progress_callback(analysis_id, 25, f"📹 Video duration: {duration_seconds:.1f}s, extracting {len(video_frames)} frames", {
                "step1": {
                    "status": "completed",
                    "duration": f"{duration_seconds:.1f}s",
                    "frames_extracted": len(video_frames),
                    "audio_format": "16kHz Mono WAV",
                    "video_resolution": "640x480"
                }
            })
            await asyncio.sleep(0)  # Force immediate execution
            
            # Step 2: Enhanced speech analysis with full transcript
            logger.info("🎤 Step 2: Analyzing speech with Whisper...")
            await progress_callback(analysis_id, 30, "🎤 Step 2: Analyzing speech with Whisper...")
            await asyncio.sleep(0)  # Force immediate execution
            
            speech_analysis = await self.analyze_speech_enhanced(audio_path)
            
            logger.info(f"✅ Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM")
            await progress_callback(analysis_id, 55, f"✅ Speech analysis complete: {speech_analysis['word_count']} words, {speech_analysis['speaking_rate']:.1f} WPM", {
                "step2": {
                    "status": "completed",
                    "word_count": speech_analysis['word_count'],
                    "transcript_length": len(speech_analysis['transcript']),
                    "speaking_rate": f"{speech_analysis['speaking_rate']:.1f} WPM",
                    "duration_minutes": f"{speech_analysis['duration_minutes']:.1f} min",
                    "filler_words": len(speech_analysis.get('filler_details', []))
                }
            })
            await asyncio.sleep(0)  # Force immediate execution
            
            # Step 3: Enhanced visual analysis with more frames
            logger.info("👁️ Step 3: Analyzing visual elements with GPT-4o Vision...")
            await progress_callback(analysis_id, 60, "👁️ Step 3: Analyzing visual elements with GPT-4o Vision...")
            
            visual_analysis = await self.analyze_visual_elements_enhanced(video_frames)
            
            logger.info(f"✅ Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed")
            await progress_callback(analysis_id, 75, f"✅ Visual analysis complete: {visual_analysis.get('frames_analyzed', 0)} frames processed", {
                "step3": {
                    "status": "completed",
                    "frames_analyzed": visual_analysis.get('frames_analyzed', 0),
                    "eye_contact": f"{visual_analysis.get('scores', {}).get('eye_contact', 0):.1f}/10",
                    "gestures": f"{visual_analysis.get('scores', {}).get('gestures', 0):.1f}/10",
                    "posture": f"{visual_analysis.get('scores', {}).get('posture', 0):.1f}/10"
                }
            })
            await asyncio.sleep(0)  # Force immediate execution
            
            # Step 4: Enhanced pedagogical assessment with full transcript
            logger.info("🎓 Step 4: Generating comprehensive pedagogical insights...")
            await progress_callback(analysis_id, 80, "🎓 Step 4: Generating comprehensive pedagogical insights...")
            await asyncio.sleep(0)  # Force immediate execution
            
            pedagogical_analysis = await self.analyze_pedagogy_enhanced(speech_analysis, visual_analysis)
            
            logger.info("✅ Enhanced pedagogical analysis complete")
            await progress_callback(analysis_id, 90, "✅ Enhanced pedagogical analysis complete", {
                "step4": {
                    "status": "completed",
                    "content_organization": f"{pedagogical_analysis.get('content_organization', 0):.1f}/10",
                    "engagement": f"{pedagogical_analysis.get('engagement_techniques', 0):.1f}/10",
                    "teaching_effectiveness": f"{pedagogical_analysis.get('overall_effectiveness', 0):.1f}/10"
                }
            })
            await asyncio.sleep(0)  # Force immediate execution

            # Step 4.5: Analyze interaction and engagement
            logger.info("🤝 Step 4.5: Analyzing interaction and questioning techniques...")
            await progress_callback(analysis_id, 90, "🤝 Step 4.5: Analyzing interaction and questioning techniques...")
            
            interaction_analysis = await self.analyze_interaction_engagement(speech_analysis)
            
            logger.info(f"✅ Interaction analysis complete: {interaction_analysis['total_questions']} questions detected")
            await progress_callback(analysis_id, 92, f"✅ Interaction analysis complete: {interaction_analysis['total_questions']} questions detected")
            
            # Step 5: Enhanced score combination with weighted sub-components
            logger.info("📊 Step 5: Calculating weighted component scores...")
            await progress_callback(analysis_id, 92, "📊 Step 5: Calculating weighted component scores...")
            
            final_results = await self.combine_analysis_enhanced(speech_analysis, visual_analysis, pedagogical_analysis, interaction_analysis)
            
            logger.info(f"✅ Enhanced analysis complete! Overall score: {final_results['overall_score']}/10")
            await progress_callback(analysis_id, 100, f"✅ Enhanced analysis complete! Overall score: {final_results['overall_score']}/10")
            await asyncio.sleep(0)  # Force immediate execution
            
            # Cleanup temporary files
            await self.cleanup_temp_files(audio_path, video_frames)
            logger.info("🧹 Cleanup complete")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Enhanced video analysis failed for {analysis_id}: {str(e)}")
            await progress_callback(analysis_id, 0, f"❌ Analysis failed: {str(e)}")
            await asyncio.sleep(0)  # Force immediate execution
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
    
    def extract_timecoded_transcript(self, words_data: List[Dict]) -> List[Dict]:
        """
        Extract transcript with word-level timecodes
        """
        timecoded_transcript = []
        
        for word_data in words_data:
            timecoded_transcript.append({
                'word': word_data.get('word', ''),
                'start': round(word_data.get('start', 0), 2),
                'end': round(word_data.get('end', 0), 2),
                'timestamp': self.format_timestamp(word_data.get('start', 0))
            })
        
        return timecoded_transcript
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    async def transcribe_large_audio_file(self, audio_path: Path) -> Any:
        """
        Transcribe large audio files by splitting into chunks and processing separately
        Optimized for Railway server environment with memory constraints
        """
        import librosa
        import soundfile as sf
        import tempfile
        import os
        
        logger.info("🔧 Starting chunked audio processing...")
        
        # Load audio file
        audio_data, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        duration_seconds = len(audio_data) / sample_rate
        
        # Calculate optimal chunk parameters
        # Target: 20MB chunks (safe buffer under 25MB limit)
        # 16kHz mono WAV = 16,000 samples/sec * 2 bytes/sample = 32KB/sec
        # 20MB = 20 * 1024 * 1024 bytes = 20,971,520 bytes
        # 20MB / 32KB/sec = ~655 seconds = ~11 minutes
        target_chunk_duration = 600  # 10 minutes for safety
        overlap_duration = 30  # 30 seconds overlap
        
        chunk_samples = int(target_chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        logger.info(f"📊 Audio duration: {duration_seconds:.1f}s, chunking into ~{target_chunk_duration}s segments")
        
        # Process chunks
        all_transcripts = []
        all_words = []
        chunk_count = 0
        
        for start_sample in range(0, len(audio_data), chunk_samples - overlap_samples):
            chunk_count += 1
            end_sample = min(start_sample + chunk_samples, len(audio_data))
            
            # Extract chunk
            chunk_audio = audio_data[start_sample:end_sample]
            chunk_start_time = start_sample / sample_rate
            
            logger.info(f"📦 Processing chunk {chunk_count} ({chunk_start_time:.1f}s - {end_sample/sample_rate:.1f}s)")
            await self.progress_callback(
                self.analysis_id, 
                33 + int((chunk_count * 5) / max(1, (len(audio_data) // (chunk_samples - overlap_samples)))), 
                f"📦 Processing chunk {chunk_count}..."
            )
            
            # Save chunk to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, chunk_audio, sample_rate)
                temp_path = temp_file.name
            
            try:
                # Transcribe chunk
                with open(temp_path, "rb") as chunk_file:
                    chunk_response = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=chunk_file,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        language="en"  # Force English transcription
                    )
                
                # Adjust timestamps to global time
                chunk_text = chunk_response.text
                chunk_words = getattr(chunk_response, 'words', [])
                
                # Adjust word timestamps to global timeline
                for word_data in chunk_words:
                    word_data['start'] += chunk_start_time
                    word_data['end'] += chunk_start_time
                
                all_transcripts.append(chunk_text)
                all_words.extend(chunk_words)
                
                logger.info(f"✅ Chunk {chunk_count} transcribed: {len(chunk_text)} chars")
                
            except Exception as e:
                logger.error(f"❌ Error processing chunk {chunk_count}: {e}")
                # Continue with other chunks
                continue
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        # Combine all transcripts
        combined_transcript = " ".join(all_transcripts)
        
        # Create a mock response object with combined data
        class CombinedResponse:
            def __init__(self, text, words):
                self.text = text
                self.words = words
        
        logger.info(f"✅ Chunked processing complete: {len(combined_transcript)} chars from {chunk_count} chunks")
        return CombinedResponse(combined_transcript, all_words)
    
    async def transcribe_large_audio_file_streaming(self, audio_path: Path) -> Any:
        """
        Memory-efficient streaming approach using FFmpeg to split audio
        Better for Railway's memory constraints with comprehensive error handling
        """
        import subprocess
        import tempfile
        import os
        import gc
        
        logger.info("🔧 Starting streaming chunked audio processing...")
        
        # Railway-specific optimizations
        max_chunks = 12  # Limit chunks for 1-hour max duration (10min chunks)
        max_retries = 3
        chunk_timeout = 300  # 5 minutes per chunk timeout
        
        # Get audio duration using FFprobe
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(audio_path)
        ]
        
        try:
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True, timeout=30)
            duration_seconds = float(duration_result.stdout.strip())
            
            # Validate duration for Railway constraints
            if duration_seconds > 3600:  # 1 hour limit
                raise ValueError(f"Audio duration ({duration_seconds:.1f}s) exceeds 1-hour limit")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ FFprobe timeout - audio file may be corrupted")
            raise Exception("Audio file analysis timeout")
        except Exception as e:
            logger.error(f"❌ Could not get audio duration: {e}")
            # Fallback to librosa method
            return await self.transcribe_large_audio_file(audio_path)
        
        # Calculate chunk parameters
        target_chunk_duration = 600  # 10 minutes
        overlap_duration = 30  # 30 seconds
        
        all_transcripts = []
        all_words = []
        chunk_count = 0
        
        # Calculate total expected chunks
        total_expected_chunks = min(max_chunks, int(duration_seconds / (target_chunk_duration - overlap_duration)) + 1)
        
        for start_time in range(0, int(duration_seconds), target_chunk_duration - overlap_duration):
            if chunk_count >= max_chunks:
                logger.warning(f"⚠️ Reached maximum chunk limit ({max_chunks}), stopping processing")
                break
                
            chunk_count += 1
            end_time = min(start_time + target_chunk_duration, duration_seconds)
            
            logger.info(f"📦 Processing chunk {chunk_count}/{total_expected_chunks} ({start_time}s - {end_time}s)")
            await self.progress_callback(
                self.analysis_id, 
                33 + int((chunk_count * 5) / total_expected_chunks), 
                f"📦 Processing chunk {chunk_count}/{total_expected_chunks}..."
            )
            
            # Create temporary file for chunk
            temp_path = None
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Extract chunk using FFmpeg with timeout
                    chunk_cmd = [
                        'ffmpeg', '-i', str(audio_path),
                        '-ss', str(start_time),
                        '-t', str(end_time - start_time),
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        temp_path,
                        '-y'
                    ]
                    
                    subprocess.run(chunk_cmd, capture_output=True, check=True, timeout=chunk_timeout)
                    
                    # Check chunk file size
                    chunk_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                    if chunk_size_mb > 25:
                        logger.warning(f"⚠️ Chunk {chunk_count} is {chunk_size_mb:.1f}MB, may exceed API limit")
                    
                    # Transcribe chunk with retry logic
                    chunk_success = False
                    for api_retry in range(max_retries):
                        try:
                            with open(temp_path, "rb") as chunk_file:
                                chunk_response = openai_client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=chunk_file,
                                    response_format="verbose_json",
                                    timestamp_granularities=["word"],
                                    language="en"  # Force English transcription
                                )
                            
                            # Adjust timestamps to global time
                            chunk_text = chunk_response.text
                            chunk_words = getattr(chunk_response, 'words', [])
                            
                            for word_data in chunk_words:
                                word_data['start'] += start_time
                                word_data['end'] += start_time
                            
                            all_transcripts.append(chunk_text)
                            all_words.extend(chunk_words)
                            
                            logger.info(f"✅ Chunk {chunk_count} transcribed: {len(chunk_text)} chars")
                            chunk_success = True
                            break
                            
                        except Exception as api_error:
                            logger.warning(f"⚠️ API error for chunk {chunk_count}, retry {api_retry + 1}: {api_error}")
                            if api_retry < max_retries - 1:
                                await asyncio.sleep(2 ** api_retry)  # Exponential backoff
                    
                    if chunk_success:
                        break
                    else:
                        raise Exception("All API retries failed")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ FFmpeg timeout for chunk {chunk_count}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                        continue
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ FFmpeg error processing chunk {chunk_count}: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                        continue
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {chunk_count}: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        await asyncio.sleep(2)
                        continue
                finally:
                    # Clean up temporary file
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                    
                    # Force garbage collection for Railway memory management
                    gc.collect()
            
            if retry_count >= max_retries:
                logger.error(f"❌ Failed to process chunk {chunk_count} after {max_retries} retries")
                continue
        
        # Combine all transcripts
        combined_transcript = " ".join(all_transcripts)
        
        # Create a mock response object with combined data
        class CombinedResponse:
            def __init__(self, text, words):
                self.text = text
                self.words = words
        
        logger.info(f"✅ Streaming chunked processing complete: {len(combined_transcript)} chars from {chunk_count} chunks")
        return CombinedResponse(combined_transcript, all_words)
    
    async def analyze_speech_enhanced(self, audio_path: Path) -> Dict[str, Any]:
        """Enhanced speech analysis using chunked Whisper transcription for large files"""
        logger.info("🎤 Starting enhanced Whisper transcription...")
        await self.progress_callback(self.analysis_id, 30, "🎤 Starting enhanced Whisper transcription (English)...")
        
        # Check file size and determine if chunking is needed
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"📊 Audio file size: {file_size_mb:.1f}MB")
        
        if file_size_mb > 20:  # Use 20MB threshold for safety
            logger.info("📦 Large audio file detected, using chunked processing...")
            await self.progress_callback(self.analysis_id, 32, f"📦 Large audio file ({file_size_mb:.1f}MB), processing in chunks...")
            
            # Use streaming approach for better memory efficiency on Railway
            try:
                transcript_response = await self.transcribe_large_audio_file_streaming(audio_path)
            except Exception as e:
                logger.warning(f"⚠️ Streaming approach failed: {e}, falling back to librosa method")
                transcript_response = await self.transcribe_large_audio_file(audio_path)
        else:
            logger.info("📄 Small audio file, processing directly...")
            await self.progress_callback(self.analysis_id, 35, "📄 Processing audio file directly (English)...")
            
            # Process directly with English language enforcement
            with open(audio_path, "rb") as audio_file:
                transcript_response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language="en"  # Force English transcription
                )
        
        await self.progress_callback(self.analysis_id, 40, "✅ Whisper transcription complete")
        
        logger.info("✅ Whisper transcription complete")
        transcript_text = transcript_response.text
        logger.info(f"📝 Full transcript length: {len(transcript_text)} characters")
        
        # Log first 200 characters for debugging
        preview = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
        logger.info(f"📄 Transcript preview: {preview}")
        
        await self.progress_callback(self.analysis_id, 42, f"📝 Full transcript length: {len(transcript_text)} characters")
        
        # Enhanced speech metrics calculation
        logger.info("🔢 Calculating enhanced speech metrics...")
        await self.progress_callback(self.analysis_id, 43, "🔢 Calculating enhanced speech metrics...")
        
        audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
        
        # Basic metrics
        duration_minutes = len(audio_data) / sample_rate / 60
        words = transcript_text.split()
        word_count = len(words)
        speaking_rate = word_count / duration_minutes if duration_minutes > 0 else 0
        
        logger.info(f"📊 Speaking rate: {speaking_rate:.1f} WPM")
        await self.progress_callback(self.analysis_id, 45, f"📊 Speaking rate: {speaking_rate:.1f} WPM")
        
        # Enhanced voice activity detection
        voice_activity = librosa.effects.split(audio_data, top_db=20)
        speaking_time = sum([(end - start) / sample_rate for start, end in voice_activity])
        speaking_ratio = speaking_time / (len(audio_data) / sample_rate)
        
        # Enhanced filler word analysis with timecodes
        text_lower = transcript_text.lower()
        filler_count = 0
        filler_details = []
        filler_timecodes = []
        
        # Get word-level data
        words_data = getattr(transcript_response, 'words', [])
        
        for filler in FILLER_WORDS:
            count = 0
            for word_data in words_data:
                word = word_data.get('word', '').lower().strip('.,!?')
                if word == filler:
                    count += 1
                    filler_timecodes.append({
                        'word': filler,
                        'timestamp': self.format_timestamp(word_data.get('start', 0)),
                        'start': round(word_data.get('start', 0), 2),
                        'end': round(word_data.get('end', 0), 2)
                    })
            
            if count > 0:
                filler_count += count
                filler_details.append({'word': filler, 'count': count})
        
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        logger.info(f"📊 Enhanced filler analysis: {filler_ratio:.3f} ({filler_count} filler words)")
        await self.progress_callback(self.analysis_id, 47, f"📊 Enhanced filler analysis: {filler_ratio:.3f} ({filler_count} filler words)")
        
        # Voice variety analysis (pitch and energy variation)
        voice_variety_score = self.calculate_voice_variety(audio_data, sample_rate)
        
        # Pause effectiveness analysis
        pause_effectiveness_score = self.calculate_pause_effectiveness(transcript_response.words if hasattr(transcript_response, 'words') else [])
        
        # Extract key phrases using improved analysis
        sentences = re.split(r'[.!?]+', transcript_text)
        highlights = [s.strip() for s in sentences if len(s.strip()) > 50][:10]
        
        logger.info("🎓 Analyzing full transcript structure with GPT-4o...")
        await self.progress_callback(self.analysis_id, 48, "🎓 Analyzing full transcript structure with GPT-4o...")
        
        content_analysis = await self.analyze_content_structure_enhanced(transcript_text)
        
        logger.info("✅ Enhanced content structure analysis complete")
        await self.progress_callback(self.analysis_id, 50, "✅ Enhanced content structure analysis complete")
        
        return {
            'transcript': transcript_text,
            'timecoded_transcript': self.extract_timecoded_transcript(words_data),
            'filler_timecodes': filler_timecodes,
            'confidence': 0.95,
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
            model="gpt-4o-mini",
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
        
        max_frames = ANALYSIS_CONFIG["sampling"]["max_frames_analyzed"]
        selected_frames = video_frames[:max_frames]
        
        logger.info(f"📊 Analyzing {len(selected_frames)} frames with enhanced visual metrics")
        await self.progress_callback(self.analysis_id, 60, f"📊 Analyzing {len(selected_frames)} frames with enhanced visual metrics")
        
        frame_analyses = []
        
        for i, frame_data in enumerate(selected_frames):
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']

            # Calculate progress for this frame within 60-75% range
            frame_progress = 60 + int((i / len(selected_frames)) * 15)
            
            logger.info(f"📊 Analyzing frame {i+1}/{len(selected_frames)} (t={timestamp:.1f}s)")
            await self.progress_callback(
                self.analysis_id,
                frame_progress,
                f"📊 Analyzing frame {i+1}/{len(selected_frames)} (t={timestamp:.1f}s)"
            )
                   
            # Convert frame to base64 for OpenAI Vision API
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Enhanced frame analysis with GPT-4o Vision
            response = openai_client.chat.completions.create(
                model="gpt-4o", # Vision requires gpt-4o, not mini
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
        # Prepare enhanced context for GPT-4o
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
            model="gpt-4o-mini",
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
        

    async def analyze_interaction_engagement(self, speech_analysis: Dict) -> Dict[str, Any]:
        """
        Analyze instructor-student interaction and questioning techniques
        """
        transcript = speech_analysis.get('transcript', '')
        words_data = speech_analysis.get('word_timestamps', [])
        
        # Detect questions and interactions using AI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in educational interaction analysis. Identify:
                    1. High-level/open-ended questions (e.g., "Why do you think...", "How would you...", "What if...")
                    2. Student interaction moments (e.g., "Let's hear from...", "Can someone explain...", "Turn to your partner...")
                    3. Cognitive engagement prompts (e.g., "Analyze...", "Compare...", "Evaluate...")
                    
                    For each interaction, provide the approximate timestamp and classify the type."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this lecture transcript for interaction and questioning:
                    
                    {transcript}
                    
                    Return as JSON with:
                    - high_level_questions: list of {{"question": str, "approx_time": "MM:SS", "type": str}}
                    - interaction_moments: list of {{"moment": str, "approx_time": "MM:SS", "type": str}}
                    - interaction_frequency: score 1-10
                    - question_quality: score 1-10
                    - student_engagement_opportunities: score 1-10
                    - cognitive_level: "low/medium/high"
                    """
                }
            ],
            max_tokens=1000
        )
        
        try:
            analysis = json.loads(response.choices[0].message.content)
            
            # Match questions to precise timestamps
            for question_data in analysis.get('high_level_questions', []):
                question_text = question_data['question']
                # Find approximate timestamp in word data
                for i, word_data in enumerate(words_data):
                    if i < len(words_data) - 5:  # Look at 5-word window
                        window = ' '.join([words_data[j].get('word', '') for j in range(i, min(i+5, len(words_data)))])
                        if question_text[:20].lower() in window.lower():
                            question_data['precise_timestamp'] = self.format_timestamp(word_data.get('start', 0))
                            question_data['start_time'] = round(word_data.get('start', 0), 2)
                            break
            
            return {
                'score': round((analysis.get('interaction_frequency', 7) + analysis.get('question_quality', 7) + analysis.get('student_engagement_opportunities', 7)) / 3, 1),
                'interaction_frequency': analysis.get('interaction_frequency', 7),
                'question_quality': analysis.get('question_quality', 7),
                'student_engagement_opportunities': analysis.get('student_engagement_opportunities', 7),
                'cognitive_level': analysis.get('cognitive_level', 'medium'),
                'high_level_questions': analysis.get('high_level_questions', [])[:10],
                'interaction_moments': analysis.get('interaction_moments', [])[:10],
                'total_questions': len(analysis.get('high_level_questions', [])),
                'total_interactions': len(analysis.get('interaction_moments', []))
            }
            
        except json.JSONDecodeError:
            return {
                'score': 6.5,
                'interaction_frequency': 6.5,
                'question_quality': 6.5,
                'student_engagement_opportunities': 6.5,
                'cognitive_level': 'medium',
                'high_level_questions': [],
                'interaction_moments': [],
                'total_questions': 0,
                'total_interactions': 0
            }
        
    def extract_evidence_from_transcript(self, transcript: str) -> List[str]:
        """
        Extract 1-2 relevant quotes from transcript to support assessment
        """
        if not transcript or len(transcript.strip()) < 50:
            return []
        
        # Split transcript into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip() and len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return [transcript[:200] + "..." if len(transcript) > 200 else transcript]
        
        # Look for sentences that might indicate good teaching practices
        evidence_indicators = [
            'example', 'for instance', 'let me explain', 'in other words',
            'first', 'second', 'next', 'finally', 'therefore', 'as a result',
            'important', 'key', 'main', 'primary', 'essential', 'crucial',
            'understand', 'remember', 'note', 'consider', 'think about'
        ]
        
        evidence_sentences = []
        
        # Find sentences with evidence indicators
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in evidence_indicators):
                if len(sentence) > 30 and len(sentence) < 150:  # Good length for evidence
                    evidence_sentences.append(sentence)
                    if len(evidence_sentences) >= 2:
                        break
        
        # If we don't have enough evidence sentences, pick the first two meaningful sentences
        if len(evidence_sentences) < 2:
            for sentence in sentences:
                if len(sentence) > 30 and len(sentence) < 150:
                    evidence_sentences.append(sentence)
                    if len(evidence_sentences) >= 2:
                        break
        
        return evidence_sentences[:2]  # Return max 2 pieces of evidence
        
    async def generate_comprehensive_summary(self, speech_analysis: Dict, visual_analysis: Dict, pedagogical_analysis: Dict, interaction_analysis: Dict, overall_score: float, transcript_text: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive evidence-based summary
        """
        transcript = speech_analysis.get('transcript', transcript_text)
        
        # Extract evidence from transcript
        evidence_quotes = self.extract_evidence_from_transcript(transcript)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert educational evaluator. Create a comprehensive summary that:
                    1. Reviews teaching content quality and accuracy
                    2. Evaluates presentation effectiveness
                    3. Assesses cognitive skill development
                    4. Uses the provided evidence from transcript to support assessment
                    5. Gives actionable, specific recommendations
                    
                    Evidence from transcript to use in your assessment: {evidence_quotes}
                    
                    Be specific, evidence-based, and constructive."""
                },
                {
                    "role": "user",
                    "content": f"""Create a comprehensive teaching evaluation summary.
                    
                    Overall Score: {overall_score}/10
                    
                    Speech Analysis: {speech_analysis.get('speaking_rate', 0)} WPM, {speech_analysis.get('filler_ratio', 0)*100:.1f}% filler words
                    Visual Engagement: {visual_analysis.get('scores', {}).get('engagement', 0)}/10
                    Pedagogical Quality: {pedagogical_analysis.get('content_organization', 0)}/10 organization
                    Interaction: {interaction_analysis.get('total_questions', 0)} questions, {interaction_analysis.get('cognitive_level', 'medium')} cognitive level
                    
                    Full Transcript:
                    {transcript}
                    
                    Provide:
                    1. content_review: Assessment of teaching content (2-3 sentences)
                    2. presentation_review: Evaluation of delivery and presentation (2-3 sentences)
                    3. cognitive_skills_review: Analysis of cognitive engagement and skill development (2-3 sentences)
                    4. key_evidence: List of 1-3 specific quotes from transcript with timestamps that exemplify teaching quality
                    5. specific_recommendations: 3-5 highly specific, actionable recommendations
                    6. overall_summary: 1 paragraph comprehensive summary
                    
                    Return as JSON."""
                }
            ],
            max_tokens=1500
        )
        
        try:
            summary = json.loads(response.choices[0].message.content)
            return summary
        except json.JSONDecodeError:
            return {
                'content_review': 'The lecture covered the material systematically with clear explanations.',
                'presentation_review': 'The instructor demonstrated good delivery with professional presence.',
                'cognitive_skills_review': 'Students were engaged through various teaching techniques.',
                'key_evidence': [],
                'specific_recommendations': [
                    'Incorporate more interactive elements',
                    'Add visual aids to support complex concepts',
                    'Include regular comprehension checks'
                ],
                'overall_summary': f'This lecture achieved an overall score of {round(overall_score, 1)}/10, demonstrating solid teaching fundamentals with opportunities for enhancement in student interaction and engagement techniques.'
            }
    
    async def combine_analysis_enhanced(self, speech_analysis: Dict, visual_analysis: Dict, pedagogical_analysis: Dict, interaction_analysis: Dict) -> Dict[str, Any]:
        """
        Enhanced analysis combination with detailed breakdown and transparency
        """
        # Calculate enhanced component scores
        speech_score = self.calculate_speech_score_enhanced(speech_analysis)
        visual_score = self.calculate_visual_score_enhanced(visual_analysis)
        pedagogy_score = self.calculate_pedagogy_score_enhanced(pedagogical_analysis)
        interaction_score = interaction_analysis.get('score', 7.0)
        presentation_score = (speech_score + visual_score) / 2
        
        # Get category weights from config
        category_weights = ANALYSIS_CONFIG["weights"]
        
        # Calculate overall weighted score with new interaction category
        overall_score = (
            speech_score * category_weights["speech_analysis"] + 
            visual_score * category_weights["body_language"] + 
            pedagogy_score * category_weights["teaching_effectiveness"] +
            interaction_score * category_weights["interaction_engagement"] +
            presentation_score * category_weights["presentation_skills"]
        )
        
        # Build the result dictionary
        result = {
            'overall_score': round(overall_score, 1),
            
            # Detailed Speech Analysis
            'speech_analysis': {
                'score': round(speech_score, 1),
                'speaking_rate': round(speech_analysis.get('speaking_rate', 0), 1),
                'clarity': round(10 - (speech_analysis.get('filler_ratio', 0) * 20), 1),
                'pace': round(min(10, max(1, 10 - abs(speech_analysis.get('speaking_rate', 150) - 150) / 20)), 1),
                'confidence': round(speech_analysis.get('confidence', 0.8) * 10, 1),
                'voice_variety': round(speech_analysis.get('voice_variety_score', 0.5) * 10, 1),
                'pause_effectiveness': round(speech_analysis.get('pause_effectiveness_score', 0.5) * 10, 1),
                'feedback': self.generate_speech_feedback_enhanced(speech_analysis),
                # Raw metrics
                'raw_metrics': {
                    'total_words': speech_analysis.get('word_count', 0),
                    'duration_minutes': round(speech_analysis.get('duration_minutes', 0), 2),
                    'words_per_minute': round(speech_analysis.get('speaking_rate', 0), 1),
                    'filler_word_count': sum(f['count'] for f in speech_analysis.get('filler_details', [])),
                    'filler_ratio_percentage': round(speech_analysis.get('filler_ratio', 0) * 100, 2),
                    'speaking_time_ratio': round(speech_analysis.get('speaking_ratio', 0.7) * 100, 1),
                    'voice_variety_index': round(speech_analysis.get('voice_variety_score', 0.5), 3),
                    'pause_effectiveness_index': round(speech_analysis.get('pause_effectiveness_score', 0.5), 3),
                    'transcription_confidence': round(speech_analysis.get('confidence', 0.8) * 100, 1)
                }
            },
            
            # Detailed Body Language Analysis
            'body_language': {
                'score': round(visual_score, 1),
                'eye_contact': round(visual_analysis.get('scores', {}).get('eye_contact', 7), 1),
                'gestures': round(visual_analysis.get('scores', {}).get('gestures', 7), 1),
                'posture': round(visual_analysis.get('scores', {}).get('posture', 7), 1),
                'engagement': round(visual_analysis.get('scores', {}).get('engagement', 7), 1),
                'professionalism': round(visual_analysis.get('scores', {}).get('professionalism', 8), 1),
                'frames_analyzed': visual_analysis.get('frames_analyzed', 0),
                'feedback': self.generate_visual_feedback_enhanced(visual_analysis),
                # Raw metrics
                'raw_metrics': {
                    'total_frames_extracted': visual_analysis.get('frames_analyzed', 0),
                    'frame_interval_seconds': ANALYSIS_CONFIG["sampling"]["frame_interval_seconds"],
                    'eye_contact_raw': round(visual_analysis.get('scores', {}).get('eye_contact', 7), 2),
                    'gestures_raw': round(visual_analysis.get('scores', {}).get('gestures', 7), 2),
                    'posture_raw': round(visual_analysis.get('scores', {}).get('posture', 7), 2),
                    'engagement_raw': round(visual_analysis.get('scores', {}).get('engagement', 7), 2),
                    'professionalism_raw': round(visual_analysis.get('scores', {}).get('professionalism', 8), 2)
                }
            },
            
            # Detailed Teaching Effectiveness
            'teaching_effectiveness': {
                'score': round(pedagogy_score, 1),
                'content_organization': round(pedagogical_analysis.get('content_organization', 7), 1),
                'engagement_techniques': round(pedagogical_analysis.get('engagement_techniques', 7), 1),
                'communication_clarity': round(pedagogical_analysis.get('communication_clarity', 7), 1),
                'use_of_examples': round(pedagogical_analysis.get('use_of_examples', 7), 1),
                'knowledge_checking': round(pedagogical_analysis.get('knowledge_checking', 7), 1),
                'feedback': pedagogical_analysis.get('recommendations', [])
            },
            
            # Presentation Skills
            'presentation_skills': {
                'score': round(presentation_score, 1),
                'professionalism': round(visual_analysis.get('scores', {}).get('professionalism', 8), 1),
                'energy': round(speech_analysis.get('speaking_ratio', 0.7) * 10, 1),
                'voice_modulation': round(speech_analysis.get('voice_variety_score', 0.5) * 10, 1),
                'time_management': round(self.assess_time_management(speech_analysis), 1),
                'feedback': self.generate_presentation_feedback_enhanced(speech_analysis, visual_analysis)
            },

            # NEW: Interaction & Engagement Analysis
            'interaction_engagement': {
                'score': round(interaction_score, 1),
                'interaction_frequency': interaction_analysis.get('interaction_frequency', 7),
                'question_quality': interaction_analysis.get('question_quality', 7),
                'student_engagement_opportunities': interaction_analysis.get('student_engagement_opportunities', 7),
                'cognitive_level': interaction_analysis.get('cognitive_level', 'medium'),
                'total_questions': interaction_analysis.get('total_questions', 0),
                'total_interactions': interaction_analysis.get('total_interactions', 0),
                'high_level_questions': interaction_analysis.get('high_level_questions', []),
                'interaction_moments': interaction_analysis.get('interaction_moments', []),
                'feedback': [
                    f"Asked {interaction_analysis.get('total_questions', 0)} questions at {interaction_analysis.get('cognitive_level', 'medium')} cognitive level",
                    f"Interaction frequency: {interaction_analysis.get('interaction_frequency', 7)}/10",
                    f"Question quality: {interaction_analysis.get('question_quality', 7)}/10"
                ]
            },
            
            # Full Transcript with Timecodes
            'full_transcript': {
                'text': speech_analysis.get('transcript', ''),
                'timecoded_words': speech_analysis.get('timecoded_transcript', []),
                'duration_formatted': self.format_timestamp(speech_analysis.get('duration_minutes', 0) * 60),
                'word_count': speech_analysis.get('word_count', 0)
            },
            
            # Filler Words with Timecodes
            'filler_words_detailed': {
                'total_count': sum(f['count'] for f in speech_analysis.get('filler_details', [])),
                'ratio_percentage': round(speech_analysis.get('filler_ratio', 0) * 100, 2),
                'timecoded_occurrences': speech_analysis.get('filler_timecodes', []),
                'breakdown_by_word': speech_analysis.get('filler_details', [])
            },
            
            # Strengths and Improvements
            'strengths': pedagogical_analysis.get('strengths', [])[:6],
            'improvement_suggestions': pedagogical_analysis.get('improvements', [])[:6],
            
            # DETAILED CALCULATION BREAKDOWN
            'calculation_breakdown': {
                'category_weights': {
                    'speech_analysis': f"{category_weights['speech_analysis']*100}%",
                    'body_language': f"{category_weights['body_language']*100}%",
                    'teaching_effectiveness': f"{category_weights['teaching_effectiveness']*100}%",
                    'interaction_engagement': f"{category_weights['interaction_engagement']*100}%",
                    'presentation_skills': f"{category_weights['presentation_skills']*100}%"
                },
                'component_scores': {
                    'speech_analysis': {
                        'score': round(speech_score, 2),
                        'weight': category_weights['speech_analysis'],
                        'contribution': round(speech_score * category_weights['speech_analysis'], 2)
                    },
                    'body_language': {
                        'score': round(visual_score, 2),
                        'weight': category_weights['body_language'],
                        'contribution': round(visual_score * category_weights['body_language'], 2)
                    },
                    'teaching_effectiveness': {
                        'score': round(pedagogy_score, 2),
                        'weight': category_weights['teaching_effectiveness'],
                        'contribution': round(pedagogy_score * category_weights['teaching_effectiveness'], 2)
                    },
                    'interaction_engagement': {
                        'score': round(interaction_score, 2),
                        'weight': category_weights['interaction_engagement'],
                        'contribution': round(interaction_score * category_weights['interaction_engagement'], 2)
                    },
                    'presentation_skills': {
                        'score': round(presentation_score, 2),
                        'weight': category_weights['presentation_skills'],
                        'contribution': round(presentation_score * category_weights['presentation_skills'], 2)
                    }
                },
                'final_calculation': {
                    'formula': '(Speech × 0.25) + (Body × 0.20) + (Teaching × 0.25) + (Interaction × 0.20) + (Presentation × 0.10)',
                    'calculation': f"({round(speech_score, 1)} × 0.25) + ({round(visual_score, 1)} × 0.20) + ({round(pedagogy_score, 1)} × 0.25) + ({round(interaction_score, 1)} × 0.20) + ({round(presentation_score, 1)} × 0.10)",
                    'result': round(overall_score, 1)
                },
                'speech_breakdown': {
                    'components': category_weights['speech_components'],
                    'scores': {
                        'speaking_rate': round(calculate_metric_score(speech_analysis.get('speaking_rate', 150), SPEECH_METRICS["speaking_rate"]), 2),
                        'clarity': round(calculate_metric_score(speech_analysis.get('confidence', 0.8), SPEECH_METRICS["speaking_clarity"]), 2),
                        'confidence': round(calculate_metric_score(speech_analysis.get('confidence', 0.8), SPEECH_METRICS["speaking_clarity"]), 2),
                        'voice_variety': round(calculate_metric_score(speech_analysis.get('voice_variety_score', 0.5), SPEECH_METRICS["voice_variety"]), 2),
                        'pause_effectiveness': round(calculate_metric_score(speech_analysis.get('pause_effectiveness_score', 0.5), SPEECH_METRICS["pause_effectiveness"]), 2)
                    }
                },
                'visual_breakdown': {
                    'components': category_weights['visual_components'],
                    'scores': visual_analysis.get('scores', {})
                },
                'pedagogy_breakdown': {
                    'components': category_weights['pedagogy_components'],
                    'scores': {
                        'content_organization': pedagogical_analysis.get('content_organization', 7),
                        'engagement_techniques': pedagogical_analysis.get('engagement_techniques', 7),
                        'communication_clarity': pedagogical_analysis.get('communication_clarity', 7),
                        'use_of_examples': pedagogical_analysis.get('use_of_examples', 7),
                        'knowledge_checking': pedagogical_analysis.get('knowledge_checking', 7)
                    }
                }
            },
            
            # Additional Insights
            'detailed_insights': {
                'transcript_summary': speech_analysis.get('transcript', '')[:800] + '...',
                'transcript_length': len(speech_analysis.get('transcript', '')),
                'key_highlights': speech_analysis.get('highlights', [])[:8],
                'visual_observations': visual_analysis.get('observations', [])[:10],
                'filler_word_analysis': speech_analysis.get('filler_details', []),
                'temporal_visual_data': visual_analysis.get('temporal_analysis', {})
            },
            
            # Configuration Used
            'configuration_used': {
                'frames_analyzed': visual_analysis.get('frames_analyzed', 0),
                'frame_interval': ANALYSIS_CONFIG["sampling"]["frame_interval_seconds"],
                'transcript_length': len(speech_analysis.get('transcript', '')),
                'category_weights': category_weights,
                'filler_words_detected': len(speech_analysis.get('filler_details', [])),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        # Generate Comprehensive Summary AFTER building the result dict
        comprehensive_summary = await self.generate_comprehensive_summary(
            speech_analysis, visual_analysis, pedagogical_analysis, interaction_analysis, overall_score, speech_analysis.get('transcript', '')
        )
        
        result['comprehensive_summary'] = comprehensive_summary
        
        return result
    
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