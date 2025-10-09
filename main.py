from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os
import shutil
from pathlib import Path
import uuid
import asyncio
from datetime import datetime 
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Import enhanced AI processor and configuration
try:
    from ai_processor import video_processor
    from metrics_config import get_configurable_parameters, update_configuration, ANALYSIS_CONFIG
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: Enhanced AI processor not available. Running in mock mode.")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, analysis_id: str):
        await websocket.accept()
        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = []
        self.active_connections[analysis_id].append(websocket)
        print(f"✅ WebSocket connected for {analysis_id}")

    def disconnect(self, websocket: WebSocket, analysis_id: str):
        if analysis_id in self.active_connections:
            self.active_connections[analysis_id].remove(websocket)
            if len(self.active_connections[analysis_id]) == 0:
                del self.active_connections[analysis_id]
        print(f"❌ WebSocket disconnected for {analysis_id}")

    async def send_update(self, analysis_id: str, message: dict):
        if analysis_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[analysis_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"⚠️ Failed to send to WebSocket: {e}")
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead in dead_connections:
                self.disconnect(dead, analysis_id)

manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(title="Enhanced Discourse Analysis API", version="3.0.0")

# CORS Configuration - Enhanced for WebSocket
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Remove any @app.middleware decorators that might interfere

@app.options("/{full_path:path}")
async def options_handler(request: Request):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Pydantic models for configuration
class ConfigurationUpdate(BaseModel):
    category_weights: Optional[Dict[str, float]] = None
    speech_components: Optional[Dict[str, float]] = None
    visual_components: Optional[Dict[str, float]] = None
    pedagogy_components: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, Any]] = None
    sampling_config: Optional[Dict[str, Any]] = None
    filler_words: Optional[List[str]] = None

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global storage for analysis results (in production, use a real database)
analysis_results = {}

# Configuration - Railway Optimized
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_VIDEO_DURATION = 3600  # 1 hour in seconds
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# Railway-specific optimizations
RAILWAY_MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB Railway default
AUDIO_CHUNK_SIZE_MB = 20  # Safe chunk size under 25MB Whisper limit
MAX_AUDIO_CHUNKS = 12  # Max chunks for 1-hour video (10min chunks)
PROCESSING_TIMEOUT = 1800  # 30 minutes total processing timeout

@app.get("/")
async def root():
    return {
        "message": "Enhanced AI-Powered Discourse Analysis API is running!", 
        "version": "3.0.0",
        "features": [
            "Enhanced frame sampling (up to 40 frames)",
            "Full transcript analysis",
            "Weighted sub-component scoring",
            "Configurable thresholds",
            "Expanded filler word detection",
            "Advanced voice variety analysis",
            "Strategic pause effectiveness scoring"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with Railway optimizations"""
    import psutil
    
    # Get memory usage
    memory_info = psutil.virtual_memory()
    memory_usage_mb = memory_info.used / (1024 * 1024)
    memory_limit_mb = RAILWAY_MEMORY_LIMIT / (1024 * 1024)
    
    return {
        "status": "healthy",
        "version": "3.0.0",
        "railway_optimized": True,
        "ai_services": {
            "openai": "configured" if os.getenv('OPENAI_API_KEY') else "missing",
            "ai_processor": "enhanced" if AI_AVAILABLE else "missing",
            "chunking_enabled": True
        },
        "configuration": {
            "max_frames": ANALYSIS_CONFIG["sampling"]["max_frames_analyzed"] if AI_AVAILABLE else "N/A",
            "frame_interval": ANALYSIS_CONFIG["sampling"]["frame_interval_seconds"] if AI_AVAILABLE else "N/A",
            "full_transcript": ANALYSIS_CONFIG["sampling"]["transcript_char_limit"] is None if AI_AVAILABLE else "N/A",
            "max_video_duration": f"{MAX_VIDEO_DURATION}s",
            "audio_chunk_size": f"{AUDIO_CHUNK_SIZE_MB}MB",
            "max_audio_chunks": MAX_AUDIO_CHUNKS
        },
        "system_resources": {
            "memory_used_mb": round(memory_usage_mb, 1),
            "memory_limit_mb": round(memory_limit_mb, 1),
            "memory_usage_percent": round((memory_usage_mb / memory_limit_mb) * 100, 1),
            "processing_timeout": f"{PROCESSING_TIMEOUT}s"
        }
    }

@app.get("/configuration")
async def get_configuration():
    """Get current configuration parameters"""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI processor not available")
    
    try:
        config = get_configurable_parameters()
        return {
            "status": "success",
            "configuration": config,
            "message": "Current configuration retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@app.post("/configuration")
async def update_configuration_endpoint(config_update: ConfigurationUpdate):
    """Update configuration parameters"""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI processor not available")
    
    try:
        # Convert Pydantic model to dict and filter None values
        update_data = {k: v for k, v in config_update.dict().items() if v is not None}
        
        # Update configuration
        success = update_configuration(update_data)
        
        if success:
            return {
                "status": "success",
                "message": "Configuration updated successfully",
                "updated_parameters": list(update_data.keys())
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@app.post("/configuration/reset")
async def reset_configuration():
    """Reset configuration to default values"""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI processor not available")
    
    try:
        # Reset to defaults by importing fresh config
        from importlib import reload
        import metrics_config
        reload(metrics_config)
        
        return {
            "status": "success",
            "message": "Configuration reset to default values"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration reset failed: {str(e)}")

@app.options("/upload-video")
async def upload_video_options():
    return {"message": "OK"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a lecture video for enhanced AI-powered analysis
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify file was saved correctly
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Get current configuration for analysis
        current_config = get_configurable_parameters() if AI_AVAILABLE else {}
        
        # Initialize analysis status with enhanced info
        analysis_results[analysis_id] = {
            "status": "processing",
            "progress": 5,
            "message": "File uploaded successfully. Starting enhanced AI analysis...",
            "log_messages": [{
                "timestamp": datetime.now().isoformat(),
                "message": "📤 File uploaded successfully. Preparing analysis...",
                "progress": 5
            }],
            "filename": file.filename,
            "file_size": file_path.stat().st_size,
            "analysis_config": {
                "max_frames": current_config.get("sampling_config", {}).get("max_frames_analyzed", 40),
                "frame_interval": current_config.get("sampling_config", {}).get("frame_interval_seconds", 6),
                "full_transcript": current_config.get("sampling_config", {}).get("use_full_transcript", True),
                "enhanced_mode": True
            }
        }

        # Immediately log initialization
        print(f"🎯 INITIALIZED with {len(analysis_results[analysis_id]['log_messages'])} messages")
        
        # Add initial log message
        analysis_results[analysis_id]["log_messages"].append({
            "timestamp": datetime.now().isoformat(),
            "message": "File uploaded successfully. Starting enhanced AI analysis...",
            "progress": 5
        })

        # Start enhanced analysis in background
        if AI_AVAILABLE:
            background_tasks.add_task(process_video_with_enhanced_ai, analysis_id, file_path)
        else:
            background_tasks.add_task(process_video_mock, analysis_id, file_path)
        
        return {
            "analysis_id": analysis_id,
            "message": "Video uploaded successfully. Enhanced AI analysis started.",
            "filename": file.filename,
            "estimated_time": "4-7 minutes" if AI_AVAILABLE else "15 seconds (mock)",
            "enhancement_features": [
                f"Analyzing up to {current_config.get('sampling_config', {}).get('max_frames_analyzed', 40)} video frames",
                "Full transcript processing",
                "Advanced voice variety analysis",
                "Strategic pause effectiveness scoring",
                "Weighted sub-component calculation"
            ]
        }
        
    except Exception as e:
        # Clean up file if something went wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not process file: {str(e)}")

@app.get("/analysis-status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Check the status of an enhanced AI analysis
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    result = analysis_results[analysis_id]

    # DEBUG: Print what we're returning
    print(f"🔍 RETURNING STATUS:")
    print(f"   - Has log_messages key: {'log_messages' in result}")
    print(f"   - Log messages count: {len(result.get('log_messages', []))}")
    if 'log_messages' in result and len(result['log_messages']) > 0:
        print(f"   - First message: {result['log_messages'][0]}")
    
    return result

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete analysis results and associated files
    """
    if analysis_id in analysis_results:
        # Try to clean up any remaining files
        for file_path in UPLOAD_DIR.glob(f"{analysis_id}_*"):
            try:
                file_path.unlink()
            except Exception:
                pass
        
        del analysis_results[analysis_id]
        return {"message": "Analysis deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Analysis not found")

# New SSE endpoint
@app.get("/stream/{analysis_id}")
async def stream_progress(analysis_id: str):
    """
    Server-Sent Events endpoint with forced immediate delivery
    """
    async def event_generator():
        last_sent_count = 0
        last_progress = -1
        max_wait = 300
        elapsed = 0
        heartbeat_counter = 0
        
        # Send initial connection with padding to force flush
        initial = json.dumps({'type': 'connected', 'data': {'message': 'Connected'}})
        yield f"data: {initial}\n\n"
        yield f": heartbeat\n\n"  # Comment line to keep connection alive
        
        while elapsed < max_wait:
            sent_something = False
            
            if analysis_id in analysis_results:
                result = analysis_results[analysis_id]
                current_log_count = len(result.get('log_messages', []))
                current_progress = result.get('progress', 0)
                current_status = result.get('status', 'processing')
                
                # Send new log messages immediately with padding
                if current_log_count > last_sent_count:
                    new_logs = result.get('log_messages', [])[last_sent_count:]
                    for log in new_logs:
                        log_data = json.dumps({'type': 'log', 'data': log})
                        yield f"data: {log_data}\n\n"
                        # Force flush with comment
                        yield f": flush\n\n"
                        sent_something = True
                    last_sent_count = current_log_count
                
                # Send status update if progress changed
                if current_progress != last_progress:
                    status_data = json.dumps({
                        'type': 'status', 
                        'data': {
                            'progress': current_progress, 
                            'message': result.get('message', ''), 
                            'status': current_status
                        }
                    })
                    yield f"data: {status_data}\n\n"
                    yield f": flush\n\n"
                    last_progress = current_progress
                    sent_something = True
                
                # Check for completion
                if current_status == 'completed':
                    complete_data = json.dumps({'type': 'complete', 'data': result})
                    yield f"data: {complete_data}\n\n"
                    break
                elif current_status == 'error':
                    error_data = json.dumps({'type': 'error', 'data': result})
                    yield f"data: {error_data}\n\n"
                    break
            
            # Send heartbeat every 3 iterations (0.9 seconds) to prevent buffering
            heartbeat_counter += 1
            if heartbeat_counter >= 3:
                yield f": heartbeat {elapsed}\n\n"
                heartbeat_counter = 0
            
            # Very short sleep for responsiveness
            await asyncio.sleep(0.3)
            elapsed += 0.3
        
        # Timeout
        if elapsed >= max_wait:
            timeout_data = json.dumps({'type': 'timeout', 'data': {'message': 'Timeout'}})
            yield f"data: {timeout_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, no-transform",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
            "Content-Type": "text/event-stream; charset=utf-8"
        }
    )

@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """
    WebSocket endpoint for real-time analysis updates
    """
    await manager.connect(websocket, analysis_id)
    try:
        # Send initial state if available
        if analysis_id in analysis_results:
            await websocket.send_json({
                "type": "status_update",
                "data": analysis_results[analysis_id]
            })
        
        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_text(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, analysis_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, analysis_id)

async def update_progress(analysis_id: str, progress: int, message: str, details: dict = None):
    """
    Helper function to update analysis progress with WebSocket broadcasting
    """
    if analysis_id in analysis_results:
        analysis_results[analysis_id]["progress"] = progress
        analysis_results[analysis_id]["message"] = message
        
        # Initialize log_messages if it doesn't exist
        if "log_messages" not in analysis_results[analysis_id]:
            analysis_results[analysis_id]["log_messages"] = []
        
        # Add message to log history
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "progress": progress
        }
        analysis_results[analysis_id]["log_messages"].append(log_entry)
        
        # Add step-specific details
        if details:
            if "step_details" not in analysis_results[analysis_id]:
                analysis_results[analysis_id]["step_details"] = {}
            analysis_results[analysis_id]["step_details"].update(details)
        
        print(f"✅ Progress: [{progress}%] {message}")
        
        # Broadcast update via WebSocket
        await manager.send_update(analysis_id, {
            "type": "progress_update",
            "data": {
                "progress": progress,
                "message": message,
                "log_entry": log_entry,
                "status": analysis_results[analysis_id]["status"]
            }
        })

async def process_video_with_enhanced_ai(analysis_id: str, file_path: Path):
    """
    Process video analysis using real AI services with detailed step tracking
    """
    try:
        print(f"🚀 BACKGROUND TASK STARTED for {analysis_id}")
        
        # Simple progress callback that updates state directly
        async def progress_callback(aid, progress, message, step_data=None):
            await update_progress(aid, progress, message, step_data)
        
        # Run the AI analysis with live progress updates
        results = await video_processor.process_video(
            video_path=file_path,
            analysis_id=analysis_id,
            progress_callback=progress_callback
        )
        
        # Update with final results
        if analysis_id in analysis_results:
            analysis_results[analysis_id].update({
                "status": "completed",
                "progress": 100,
                "message": "AI analysis completed successfully!",
                "results": results,
                "processing_time": "Real AI analysis complete"
            })
            
            # Broadcast completion via WebSocket
            await manager.send_update(analysis_id, {
                "type": "analysis_complete",
                "data": analysis_results[analysis_id]
            })
        
        # Clean up uploaded file
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up file {file_path}: {e}")
            
    except Exception as e:
        print(f"Analysis failed for {analysis_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if analysis_id in analysis_results:
            analysis_results[analysis_id].update({
                "status": "error",
                "progress": 0,
                "message": f"Analysis failed: {str(e)}",
                "error_details": str(e)
            })
        
        # Clean up file on error
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

async def process_video_mock_enhanced(analysis_id: str, file_path: Path):
    """
    Enhanced mock processing with realistic progress updates
    """
    try:
        # Initialize progress tracking
        analysis_results[analysis_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Extracting enhanced audio and video components..."
        }
        
        # Simulate enhanced processing steps
        await asyncio.sleep(3)
        analysis_results[analysis_id]["progress"] = 25
        analysis_results[analysis_id]["message"] = "Analyzing 40 video frames with GPT-4 Vision..."
        
        await asyncio.sleep(4)
        analysis_results[analysis_id]["progress"] = 45
        analysis_results[analysis_id]["message"] = "Processing full transcript with advanced speech metrics..."
        
        await asyncio.sleep(4)
        analysis_results[analysis_id]["progress"] = 65
        analysis_results[analysis_id]["message"] = "Calculating voice variety and pause effectiveness..."
        
        await asyncio.sleep(3)
        analysis_results[analysis_id]["progress"] = 80
        analysis_results[analysis_id]["message"] = "Generating weighted pedagogical assessment..."
        
        await asyncio.sleep(2)
        analysis_results[analysis_id]["progress"] = 95
        analysis_results[analysis_id]["message"] = "Finalizing enhanced analysis report..."
        
        await asyncio.sleep(1)
        
        # Create enhanced mock analysis results
        final_results = create_enhanced_mock_results()
        
        analysis_results[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Enhanced analysis completed successfully! (Mock mode)",
            "results": final_results,
            "analysis_summary": {
                "frames_analyzed": 40,
                "transcript_length": 15000,
                "filler_words_detected": 8,
                "enhancement_features_used": [
                    "Enhanced frame sampling (40 frames)",
                    "Full transcript analysis",
                    "Advanced voice variety analysis",
                    "Strategic pause effectiveness",
                    "Weighted sub-component scoring"
                ]
            }
        }
        
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()
            
    except Exception as e:
        analysis_results[analysis_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Enhanced analysis failed: {str(e)}"
        }

def create_enhanced_mock_results() -> Dict[str, Any]:
    """
    Create enhanced mock analysis results with all new features
    """
    return {
        "overall_score": 8.1,
        "speech_analysis": {
            "score": 8.4,
            "speaking_rate": 165.3,
            "clarity": 8.7,
            "pace": 8.2,
            "confidence": 9.1,
            "voice_variety": 7.8,
            "pause_effectiveness": 8.0,
            "feedback": [
                "Excellent speaking rate within optimal range (165 WPM)",
                "Strong voice modulation enhances engagement",
                "Effective use of strategic pauses for emphasis",
                "Minimal filler words detected (2.1% ratio)"
            ]
        },
        "body_language": {
            "score": 7.8,
            "eye_contact": 8.2,
            "gestures": 7.5,
            "posture": 8.0,
            "engagement": 7.8,
            "professionalism": 8.5,
            "frames_analyzed": 40,
            "feedback": [
                "Consistent eye contact throughout 40 analyzed frames",
                "Natural hand gestures support content delivery",
                "Professional appearance and confident posture",
                "Visual engagement shows gradual improvement over time"
            ]
        },
        "teaching_effectiveness": {
            "score": 8.0,
            "content_organization": 8.3,
            "engagement_techniques": 7.6,
            "communication_clarity": 8.2,
            "use_of_examples": 7.8,
            "knowledge_checking": 7.4,
            "feedback": [
                "Excellent content structure with clear transitions",
                "Good use of relevant examples throughout",
                "Clear explanations of complex concepts",
                "Could incorporate more comprehension checks"
            ]
        },
        "presentation_skills": {
            "score": 8.2,
            "professionalism": 8.5,
            "energy": 8.0,
            "voice_modulation": 7.8,
            "time_management": 8.1,
            "feedback": [
                "High professional standards maintained",
                "Excellent energy and enthusiasm",
                "Good voice modulation for emphasis",
                "Effective time management throughout session"
            ]
        },
        "improvement_suggestions": [
            "Incorporate more interactive questioning techniques",
            "Add periodic comprehension checks every 10-15 minutes",
            "Consider more varied hand gestures for emphasis",
            "Experiment with strategic movement around teaching space",
            "Develop more concrete real-world examples"
        ],
        "strengths": [
            "Exceptional speaking clarity and articulation",
            "Professional and confident presentation style",
            "Well-organized content with logical progression",
            "Effective use of voice variety and pacing",
            "Strong visual presence with good eye contact",
            "Comprehensive coverage of topic material"
        ],
        "detailed_insights": {
            "transcript_summary": "This enhanced analysis processed the complete lecture transcript, analyzing voice patterns, pause effectiveness, and comprehensive content structure...",
            "key_highlights": [
                "Opening with clear learning objectives",
                "Effective transition between main topics",
                "Good use of analogy to explain complex concepts",
                "Strong conclusion with key takeaways"
            ],
            "filler_word_analysis": [
                {"word": "um", "count": 5},
                {"word": "like", "count": 3},
                {"word": "you know", "count": 2}
            ]
        },
        "configuration_used": {
            "frames_analyzed": 40,
            "transcript_length": 15247,
            "filler_words_detected": 8,
            "category_weights": {
                "speech_analysis": 30,
                "body_language": 25,
                "teaching_effectiveness": 35,
                "presentation_skills": 10
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        ws_ping_interval=20,  # Add WebSocket keep-alive
        ws_ping_timeout=20
    )