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

# CORS Configuration - Enhanced for all origins
# Note: allow_credentials must be False when allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicit methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)

# Global OPTIONS handler for preflight requests
@app.options("/{full_path:path}")
async def options_handler(request: Request):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
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
# Global storage for running processes
running_processes = {}
# Job queue system
job_queue = []
MAX_CONCURRENT_JOBS = 1  # Railway can handle 1 job at a time

# Configuration - Railway Optimized
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
MAX_VIDEO_DURATION = 7200  # 2 hours in seconds
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# Railway-specific optimizations
RAILWAY_MEMORY_LIMIT = 512 * 1024 * 1024  # 512MB Railway default
AUDIO_CHUNK_SIZE_MB = 20  # Safe chunk size under 25MB Whisper limit
MAX_AUDIO_CHUNKS = 12  # Max chunks for 1-hour video (10min chunks)
PROCESSING_TIMEOUT = 1800  # 30 minutes total processing timeout

@app.get("/")
async def root():
    return {
        "message": "MARS - Multimodal AI Reflection System API is running!", 
        "version": "3.0.0",
        "app_name": "MARS",
        "full_name": "Multimodal AI Reflection System",
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

@app.get("/deployment-info")
async def deployment_info():
    """
    Get deployment information including last deployment time
    """
    import pytz
    from pathlib import Path
    import subprocess
    
    deployment_time = None
    
    # Try to get from environment variable (set during deployment)
    if os.getenv("DEPLOYMENT_TIME"):
        deployment_time = os.getenv("DEPLOYMENT_TIME")
    else:
        # Try to get from git commit time (most accurate)
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ci', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2,
                cwd=Path(__file__).parent
            )
            if result.returncode == 0 and result.stdout.strip():
                # Git format: 2024-11-19 14:30:45 +0800
                git_time_str = result.stdout.strip()
                deployment_time = git_time_str
        except:
            pass
        
        # Fallback: use main.py modification time as proxy for deployment
        if not deployment_time:
            try:
                main_file = Path(__file__)
                if main_file.exists():
                    mtime = main_file.stat().st_mtime
                    deployment_time = datetime.fromtimestamp(mtime, tz=pytz.UTC).isoformat()
            except:
                pass
    
    # If still no time, use current time
    if not deployment_time:
        deployment_time = datetime.now(pytz.UTC).isoformat()
    
    # Convert to Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    
    # Parse deployment time
    if isinstance(deployment_time, str):
        # Try parsing git format first (2024-11-19 14:30:45 +0800)
        if ' ' in deployment_time and '+' in deployment_time or '-' in deployment_time[-6:]:
            try:
                # Git format: "2024-11-19 14:30:45 +0800"
                parts = deployment_time.split()
                if len(parts) >= 3:
                    date_part = parts[0]
                    time_part = parts[1]
                    tz_offset = parts[2]
                    dt_str = f"{date_part}T{time_part}{tz_offset[:3]}:{tz_offset[3:]}"
                    dt = datetime.fromisoformat(dt_str)
                    # Convert to UTC first, then to Singapore
                    dt = dt.astimezone(pytz.UTC)
                else:
                    dt = datetime.fromisoformat(deployment_time.replace('Z', '+00:00'))
            except:
                dt = datetime.fromisoformat(deployment_time.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(deployment_time.replace('Z', '+00:00'))
    else:
        dt = deployment_time
    
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    
    singapore_time = dt.astimezone(singapore_tz)
    
    return {
        "deployment_time": singapore_time.isoformat(),
        "deployment_time_formatted": singapore_time.strftime("%d %B %Y, %H:%M:%S"),
        "timezone": "Asia/Singapore"
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

def get_queue_status(current_user_ip: str = None):
    """
    Get current queue status and estimated wait time with concurrent user warnings
    Filters out the current user's IP to avoid false warnings
    """
    active_jobs = len([pid for pid in running_processes.keys() if analysis_results.get(pid, {}).get('status') == 'processing'])
    queued_jobs = len(job_queue)
    
    # Collect IP addresses of active and queued users (excluding current user)
    active_ips = []
    queued_ips = []
    
    # Get IPs from currently processing jobs (excluding current user)
    for analysis_id, result in analysis_results.items():
        if result.get('status') == 'processing':
            ip = result.get('client_ip', 'Unknown')
            if ip and ip != 'Unknown' and ip != current_user_ip:
                active_ips.append(ip)
    
    # Get IPs from queued jobs (excluding current user)
    for job in job_queue:
        ip = job.get('client_ip', 'Unknown')
        if ip and ip != 'Unknown' and ip != current_user_ip:
            queued_ips.append(ip)
    
    # Total unique users (active + queued, excluding current user)
    all_ips = list(set(active_ips + queued_ips))
    total_users = len(all_ips)
    
    # Estimate wait time based on current job progress
    estimated_wait_minutes = 0
    current_job_progress = 0
    
    if active_jobs > 0:
        # Find the currently running job
        for analysis_id, result in analysis_results.items():
            if result.get('status') == 'processing':
                progress = result.get('progress', 0)
                current_job_progress = progress
                # Estimate remaining time based on progress
                # Assume total processing time is 10-15 minutes for a typical video
                estimated_total_minutes = 12
                remaining_progress = 100 - progress
                estimated_wait_minutes = (remaining_progress / 100) * estimated_total_minutes
                break
    
    # Add time for queued jobs (assume 12 minutes per job)
    estimated_wait_minutes += queued_jobs * 12
    
    # Determine warning level (only if there are OTHER users, not the current user)
    warning_level = "none"
    warning_message = ""
    
    # Only show warnings if there are OTHER users (not counting current user's own jobs)
    if len(active_ips) > 0 and current_job_progress < 50:
        warning_level = "high"
        warning_message = "⚠️ Another user is currently processing a video. Processing will be very slow. We recommend waiting and trying again later."
    elif len(active_ips) > 0 and current_job_progress < 80:
        warning_level = "medium"
        warning_message = "⚠️ Another user is processing a video. Processing may be slower than usual."
    elif len(queued_ips) > 0:
        warning_level = "low"
        warning_message = "ℹ️ There are videos in the queue. Processing may take longer than usual."
    
    return {
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "estimated_wait_minutes": round(estimated_wait_minutes, 1),
        "can_start_immediately": active_jobs < MAX_CONCURRENT_JOBS,
        "warning_level": warning_level,
        "warning_message": warning_message,
        "current_job_progress": current_job_progress,
        "active_ips": active_ips,
        "queued_ips": queued_ips,
        "total_users": total_users,
        "all_ips": all_ips
    }

@app.get("/queue-status")
async def queue_status(request: Request, current_user_ip: Optional[str] = None):
    """
    Get current queue status and estimated wait time
    Filters out the current user's IP to avoid false warnings
    """
    # Get client IP from request if not provided as query parameter
    if not current_user_ip:
        current_user_ip = request.client.host if request.client else None
        # Try to get real IP from headers (for proxies/load balancers)
        if "x-forwarded-for" in request.headers:
            current_user_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
        elif "x-real-ip" in request.headers:
            current_user_ip = request.headers["x-real-ip"]
    
    # Also check query parameter for client IP (frontend can pass it)
    query_ip = request.query_params.get("client_ip")
    if query_ip:
        current_user_ip = query_ip
    
    status = get_queue_status(current_user_ip)
    # Add explicit CORS headers as backup
    response = JSONResponse(content=status)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.get("/queue-list")
async def queue_list():
    """
    Get detailed queue list with all jobs and their status
    """
    queue_list_data = []
    
    # Add currently processing job
    for analysis_id, result in analysis_results.items():
        if result.get('status') == 'processing':
            queue_list_data.append({
                "analysis_id": analysis_id,
                "filename": result.get('filename', 'Unknown'),
                "client_ip": result.get('client_ip', 'Unknown'),
                "status": "processing",
                "progress": result.get('progress', 0),
                "queued_at": result.get('queued_at', ''),
                "file_size": result.get('file_size', 0)
            })
            break
    
    # Add queued jobs
    for idx, job in enumerate(job_queue):
        # Get current status from analysis_results if available
        job_status = "queued"
        job_progress = 0
        if job["analysis_id"] in analysis_results:
            result = analysis_results[job["analysis_id"]]
            job_status = result.get('status', 'queued')
            job_progress = result.get('progress', 0)
        
        queue_list_data.append({
            "analysis_id": job["analysis_id"],
            "filename": job["filename"],
            "client_ip": job.get("client_ip", "Unknown"),
            "status": job_status,
            "progress": job_progress,
            "queued_at": job["queued_at"],
            "file_size": job.get("file_size", 0),
            "queue_position": idx + 1 + (1 if len([r for r in analysis_results.values() if r.get('status') == 'processing']) > 0 else 0)
        })
    
    response = JSONResponse(content={
        "queue": queue_list_data,
        "total_in_queue": len(queue_list_data),
        "currently_processing": len([r for r in analysis_results.values() if r.get('status') == 'processing'])
    })
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.post("/validate-passkey")
async def validate_passkey(request: Request, passkey_data: dict):
    """
    Validate the passkey before allowing video upload
    """
    expected_passkey = os.getenv('MARS_PASSKEY', '')
    
    if not expected_passkey:
        # If no passkey is set, allow access (backward compatibility)
        return JSONResponse(content={'valid': True, 'message': 'Passkey not configured'})
    
    provided_passkey = passkey_data.get('passkey', '')
    
    if provided_passkey == expected_passkey:
        return JSONResponse(content={'valid': True, 'message': 'Passkey validated'})
    else:
        return JSONResponse(
            status_code=401,
            content={'valid': False, 'message': 'Invalid passkey'}
        )

@app.post("/upload-video")
async def upload_video(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload a lecture video for enhanced AI-powered analysis with queue management
    """
    # Validate passkey if configured
    expected_passkey = os.getenv('MARS_PASSKEY', '')
    if expected_passkey:
        # Get passkey from form data (multipart/form-data)
        try:
            form_data = await request.form()
            provided_passkey = form_data.get('passkey', '')
        except:
            provided_passkey = None
        
        if not provided_passkey or provided_passkey != expected_passkey:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing passkey. Please verify your passkey."
            )
    
    # Get client IP address
    client_ip = request.client.host if request.client else "unknown"
    # Try to get real IP from headers (for proxies/load balancers)
    if "x-forwarded-for" in request.headers:
        client_ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    elif "x-real-ip" in request.headers:
        client_ip = request.headers["x-real-ip"]
    
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
    
    # Save the uploaded file (always save, even if queued)
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify file was saved correctly
        if not file_path.exists():
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Check queue status
    queue_status = get_queue_status()
    
    if not queue_status["can_start_immediately"]:
        # Add to queue
        import pytz
        singapore_tz = pytz.timezone('Asia/Singapore')
        singapore_time = datetime.now().astimezone(singapore_tz)
        
        job_queue.append({
            "analysis_id": analysis_id,
            "filename": file.filename,
            "file_size": file.size,
            "client_ip": client_ip,
            "queued_at": singapore_time.isoformat(),
            "estimated_wait_minutes": queue_status["estimated_wait_minutes"],
            "file_path": str(file_path)
        })
        
        # Initialize analysis result as queued
        analysis_results[analysis_id] = {
            "status": "queued",
            "progress": 0,
            "message": f"Video queued for processing. Estimated wait: {queue_status['estimated_wait_minutes']} minutes",
            "log_messages": [{
                "timestamp": singapore_time.isoformat(),
                "message": f"📋 Video queued for processing. Position: {len(job_queue)} in queue",
                "progress": 0
            }],
            "filename": file.filename,
            "file_size": file.size,
            "client_ip": client_ip,
            "queued_at": singapore_time.isoformat(),
            "estimated_wait_minutes": queue_status["estimated_wait_minutes"]
        }
        
        return {
            "analysis_id": analysis_id,
            "status": "queued",
            "message": f"Video queued for processing. Estimated wait: {queue_status['estimated_wait_minutes']} minutes",
            "queue_position": len(job_queue),
            "estimated_wait_minutes": queue_status["estimated_wait_minutes"]
        }
    
    # Immediate processing (not queued)
    # Get current configuration for analysis
    current_config = get_configurable_parameters() if AI_AVAILABLE else {}
    
    # Initialize analysis status with enhanced info
    # Initialize analysis result with Singapore time
    import pytz
    singapore_tz = pytz.timezone('Asia/Singapore')
    singapore_time = datetime.now().astimezone(singapore_tz)
    
    analysis_results[analysis_id] = {
        "status": "processing",
        "progress": 5,
        "message": "File uploaded successfully. Starting enhanced AI analysis...",
        "log_messages": [{
            "timestamp": singapore_time.isoformat(),
            "message": "📤 File uploaded successfully. Preparing analysis...",
            "progress": 5
        }],
        "filename": file.filename,
        "file_size": file_path.stat().st_size,
        "client_ip": client_ip,
        "analysis_config": {
            "max_frames": current_config.get("sampling_config", {}).get("max_frames_analyzed", 40),
            "frame_interval": current_config.get("sampling_config", {}).get("frame_interval_seconds", 6),
            "full_transcript": current_config.get("sampling_config", {}).get("use_full_transcript", True),
            "enhanced_mode": True
        }
    }

    # Immediately log initialization
    print(f"🎯 INITIALIZED with {len(analysis_results[analysis_id]['log_messages'])} messages")
    
    # Add initial log message with Singapore time
    import pytz
    singapore_tz = pytz.timezone('Asia/Singapore')
    singapore_time = datetime.now().astimezone(singapore_tz)
    analysis_results[analysis_id]["log_messages"].append({
        "timestamp": singapore_time.isoformat(),
        "message": "File uploaded successfully. Starting enhanced AI analysis...",
        "progress": 5
    })

    # Start enhanced analysis in background
    if AI_AVAILABLE:
        background_tasks.add_task(process_video_with_enhanced_ai, analysis_id, file_path)
    else:
        background_tasks.add_task(process_video_mock_enhanced, analysis_id, file_path)
    
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

@app.post("/stop-analysis/{analysis_id}")
async def stop_analysis(analysis_id: str):
    """
    Stop a running analysis process
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    try:
        # Mark analysis as stopped
        analysis_results[analysis_id]["status"] = "stopped"
        analysis_results[analysis_id]["message"] = "Analysis stopped by user"
        
        # Store the stop request timestamp with Singapore time
        import pytz
        singapore_tz = pytz.timezone('Asia/Singapore')
        singapore_time = datetime.now().astimezone(singapore_tz)
        analysis_results[analysis_id]["stopped_at"] = singapore_time.isoformat()
        
        # If there's a running process, mark it for termination
        if analysis_id in running_processes:
            running_processes[analysis_id]["should_stop"] = True
        
        print(f"🛑 Analysis {analysis_id} marked for stopping")
        
        return {
            "success": True,
            "message": "Analysis stop request received",
            "analysis_id": analysis_id
        }
        
    except Exception as e:
        print(f"❌ Error stopping analysis {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop analysis: {str(e)}")

@app.post("/cancel-upload/{analysis_id}")
async def cancel_upload(analysis_id: str):
    """
    Cancel an upload and clean up any partial files
    """
    try:
        # Remove from queue if present
        job_queue[:] = [job for job in job_queue if job.get("analysis_id") != analysis_id]
        
        # Clean up any uploaded files
        for file_path in UPLOAD_DIR.glob(f"{analysis_id}_*"):
            try:
                file_path.unlink()
                print(f"🗑️ Deleted partial upload: {file_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}: {e}")
        
        # Remove from analysis results if present
        if analysis_id in analysis_results:
            analysis_results[analysis_id]["status"] = "cancelled"
            analysis_results[analysis_id]["message"] = "Upload cancelled by user"
        
        # Remove from running processes if present
        if analysis_id in running_processes:
            running_processes[analysis_id]["should_stop"] = True
            del running_processes[analysis_id]
        
        print(f"🛑 Upload {analysis_id} cancelled and cleaned up")
        
        return {
            "success": True,
            "message": "Upload cancelled successfully",
            "analysis_id": analysis_id
        }
        
    except Exception as e:
        print(f"❌ Error cancelling upload {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel upload: {str(e)}")

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
        
        # Remove from running processes
        if analysis_id in running_processes:
            del running_processes[analysis_id]
        
        del analysis_results[analysis_id]
        return {"message": "Analysis deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Analysis not found")

# New SSE endpoint
@app.post("/generate-pdf-summary")
async def generate_pdf_summary(request: Request, summary_data: dict):
    """
    Generate personalized PDF summary with strengths, improvements, and evidence
    """
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI processor not available")
    
    try:
        from ai_processor import video_processor
        from openai import OpenAI
        import json
        
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Extract data from request
        overall_score = summary_data.get('overall_score', 0)
        speech_score = summary_data.get('speech_score', 0)
        body_language_score = summary_data.get('body_language_score', 0)
        teaching_effectiveness_score = summary_data.get('teaching_effectiveness_score', 0)
        interaction_score = summary_data.get('interaction_score', 0)
        presentation_score = summary_data.get('presentation_score', 0)
        high_level_questions = summary_data.get('high_level_questions', [])
        total_questions = summary_data.get('total_questions', 0)
        transcript_excerpt = summary_data.get('transcript_excerpt', '')
        sample_frames_count = summary_data.get('sample_frames_count', 0)
        filler_words = summary_data.get('filler_words', [])
        explanations = summary_data.get('explanations', {})
        
        # Prepare high-level questions text
        questions_text = ""
        if high_level_questions and len(high_level_questions) > 0:
            questions_text = "\n\nHigh-Level Questions Asked:\n"
            for idx, q in enumerate(high_level_questions[:5], 1):
                question_text = q.get('question', q.get('text', ''))
                timestamp = q.get('precise_timestamp', q.get('timestamp', ''))
                questions_text += f"{idx}. [{timestamp}] {question_text}\n"
        
        # Prepare filler words text
        filler_text = ""
        if filler_words and len(filler_words) > 0:
            filler_text = f"\n\nFiller Words Detected: {', '.join([f['word'] for f in filler_words[:5]])}"
        
        # Find strongest category
        scores = {
            'Speech Analysis': speech_score,
            'Body Language': body_language_score,
            'Teaching Effectiveness': teaching_effectiveness_score,
            'Interaction & Engagement': interaction_score,
            'Presentation Skills': presentation_score
        }
        strongest_category = max(scores.items(), key=lambda x: x[1])
        
        # Find weakest categories (for improvements)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        weakest_categories = sorted_scores[:2] if len(sorted_scores) >= 2 else sorted_scores
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert educational evaluator creating a personalized feedback summary for a lecture analysis report. Your task is to:

1. Write a personalized feedback paragraph (80-100 words) that:
   - References the overall score and key findings
   - Mentions high-level questions as evidence of analytical thinking promotion
   - Includes at least one actual question from the list provided
   - Is encouraging yet constructive

2. Identify ONE strongest strength with:
   - Clear title
   - Description explaining why it's a strength
   - Evidence from transcript or analysis (quote or specific metric)

3. Identify 1-2 areas for improvement with:
   - Area name
   - Specific recommendation
   - Evidence from scoring/metrics

Be specific, evidence-based, and professional. Use British English spelling."""
                },
                {
                    "role": "user",
                    "content": f"""Generate a personalized PDF summary for this lecture analysis.

Overall Score: {overall_score}/10

Category Scores:
- Speech Analysis: {speech_score}/10
- Body Language: {body_language_score}/10
- Teaching Effectiveness: {teaching_effectiveness_score}/10
- Interaction & Engagement: {interaction_score}/10
- Presentation Skills: {presentation_score}/10

Strongest Category: {strongest_category[0]} ({strongest_category[1]}/10)
Weakest Categories: {weakest_categories[0][0]} ({weakest_categories[0][1]}/10){f', {weakest_categories[1][0]} ({weakest_categories[1][1]}/10)' if len(weakest_categories) > 1 else ''}

Total Questions Asked: {total_questions}
{questions_text}

Transcript Excerpt (first 2000 chars):
{transcript_excerpt[:2000]}

Sample Frames Extracted: {sample_frames_count}
{filler_text}

Return as JSON with:
- personalized_feedback: string (80-100 words paragraph)
- strongest_strength: {{"title": string, "description": string, "evidence": string}}
- improvements: [{{"area": string, "description": string, "evidence": string}}, ...] (1-2 items)

Ensure the personalized_feedback includes at least one high-level question as evidence."""
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        try:
            summary_content = response.choices[0].message.content
            if not summary_content:
                raise ValueError("AI response content is None or empty")
            
            summary = json.loads(summary_content)
            
            # Ensure all required fields exist
            if 'personalized_feedback' not in summary:
                summary['personalized_feedback'] = f"This lecture achieved an overall score of {overall_score}/10, demonstrating effective teaching techniques."
            
            if 'strongest_strength' not in summary:
                summary['strongest_strength'] = {
                    'title': strongest_category[0],
                    'description': f'Strong performance in {strongest_category[0].lower()} with a score of {strongest_category[1]}/10.',
                    'evidence': f'Score of {strongest_category[1]}/10 in {strongest_category[0]}'
                }
            
            if 'improvements' not in summary or len(summary['improvements']) == 0:
                summary['improvements'] = [{
                    'area': weakest_categories[0][0],
                    'description': f'Consider focusing on improving {weakest_categories[0][0].lower()} which scored {weakest_categories[0][1]}/10.',
                    'evidence': f'Current score: {weakest_categories[0][1]}/10'
                }]
            
            return JSONResponse(content={'summary': summary})
            
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            # Fallback summary
            fallback_summary = {
                'personalized_feedback': f"This lecture achieved an overall score of {overall_score}/10. The instructor demonstrated effective teaching techniques with {total_questions} questions asked throughout the session{f', including high-level questions that promote analytical thinking' if high_level_questions else ''}.",
                'strongest_strength': {
                    'title': strongest_category[0],
                    'description': f'Strong performance in {strongest_category[0].lower()} with a score of {strongest_category[1]}/10.',
                    'evidence': f'Score of {strongest_category[1]}/10 in {strongest_category[0]}'
                },
                'improvements': [{
                    'area': weakest_categories[0][0],
                    'description': f'Consider focusing on improving {weakest_categories[0][0].lower()} which scored {weakest_categories[0][1]}/10.',
                    'evidence': f'Current score: {weakest_categories[0][1]}/10'
                }]
            }
            if len(weakest_categories) > 1:
                fallback_summary['improvements'].append({
                    'area': weakest_categories[1][0],
                    'description': f'Additionally, {weakest_categories[1][0].lower()} could be enhanced (score: {weakest_categories[1][1]}/10).',
                    'evidence': f'Current score: {weakest_categories[1][1]}/10'
                })
            
            return JSONResponse(content={'summary': fallback_summary})
            
    except Exception as e:
        print(f"Error generating PDF summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF summary: {str(e)}")

@app.get("/stream/{analysis_id}")
async def stream_progress(analysis_id: str):
    """
    Server-Sent Events endpoint with forced immediate delivery
    """
    async def event_generator():
        last_sent_count = 0
        last_progress = -1
        last_update_timestamp = None
        last_update_counter = 0
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
                current_timestamp = result.get('last_update')
                current_counter = result.get('update_counter', 0)
                
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
                
                # Send status update if ANY change detected
                current_message = result.get('message', '')
                last_message = result.get('last_message', '')
                
                if (current_progress != last_progress or 
                    current_timestamp != last_update_timestamp or
                    current_message != last_message or
                    current_counter != last_update_counter):
                    status_data = json.dumps({
                        'type': 'status', 
                        'data': {
                            'progress': current_progress, 
                            'message': current_message, 
                            'status': current_status,
                            'timestamp': current_timestamp
                        }
                    })
                    yield f"data: {status_data}\n\n"
                    yield f": flush\n\n"
                    yield f"id: {int(datetime.now().timestamp() * 1000)}\n\n"  # Force immediate delivery
                    last_progress = current_progress
                    last_update_timestamp = current_timestamp
                    last_update_counter = current_counter
                    result['last_message'] = current_message  # Store last message
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
            
            # Send heartbeat every 2 iterations (0.6 seconds) to prevent buffering
            heartbeat_counter += 1
            if heartbeat_counter >= 2:
                yield f": heartbeat {elapsed}\n\n"
                heartbeat_counter = 0
            
            # Ultra-short sleep for maximum responsiveness
            await asyncio.sleep(0.05)
            elapsed += 0.05
        
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

# Global throttling for progress updates
progress_update_times = {}

async def update_progress(analysis_id: str, progress: int, message: str, details: dict = None):
    """
    Helper function to update analysis progress with throttled SSE broadcasting
    """
    if analysis_id in analysis_results:
        # Minimal throttling - only skip if EXACTLY the same update within 10ms
        current_time = datetime.now()
        last_update_time = progress_update_times.get(analysis_id)
        
        if (last_update_time and 
            (current_time - last_update_time).total_seconds() < 0.01 and
            analysis_results[analysis_id].get("progress", 0) == progress and
            analysis_results[analysis_id].get("message", "") == message):
            # Skip only if it's the exact same update within 10ms
            return
        
        progress_update_times[analysis_id] = current_time
        
        # Update the main progress
        analysis_results[analysis_id]["progress"] = progress
        analysis_results[analysis_id]["message"] = message
        
        # Initialize log_messages if it doesn't exist
        if "log_messages" not in analysis_results[analysis_id]:
            analysis_results[analysis_id]["log_messages"] = []
        
        # Add message to log history with Singapore time
        import pytz
        singapore_tz = pytz.timezone('Asia/Singapore')
        singapore_time = current_time.astimezone(singapore_tz)
        
        log_entry = {
            "timestamp": singapore_time.isoformat(),
            "message": message,
            "progress": progress
        }
        analysis_results[analysis_id]["log_messages"].append(log_entry)
        
        # Add step-specific details
        if details:
            if "step_details" not in analysis_results[analysis_id]:
                analysis_results[analysis_id]["step_details"] = {}
            analysis_results[analysis_id]["step_details"].update(details)
        
        # Removed print statement to reduce logging rate (was causing Railway rate limit issues)
        # print(f"✅ Progress: [{progress}%] {message}")
        
        # Force immediate update by adding a timestamp and counter to trigger SSE
        # Convert to Singapore time
        import pytz
        singapore_tz = pytz.timezone('Asia/Singapore')
        singapore_time = current_time.astimezone(singapore_tz)
        analysis_results[analysis_id]["last_update"] = singapore_time.isoformat()
        analysis_results[analysis_id]["update_counter"] = analysis_results[analysis_id].get("update_counter", 0) + 1
        
        # Add a small delay to ensure the update is processed by SSE
        await asyncio.sleep(0.001)
        
        # Broadcast update via WebSocket (for any WebSocket connections)
        await manager.send_update(analysis_id, {
            "type": "progress_update",
            "data": {
                "progress": progress,
                "message": message,
                "log_entry": log_entry,
                "status": analysis_results[analysis_id]["status"],
                "timestamp": analysis_results[analysis_id]["last_update"]
            }
        })

async def process_video_with_enhanced_ai(analysis_id: str, file_path: Path):
    """
    Process video analysis using real AI services with detailed step tracking
    """
    import pytz
    try:
        print(f"🚀 BACKGROUND TASK STARTED for {analysis_id}")
        
        # Register this process for potential stopping
        singapore_tz = pytz.timezone('Asia/Singapore')
        running_processes[analysis_id] = {
            "should_stop": False,
            "started_at": datetime.now().astimezone(singapore_tz).isoformat()
        }
        
        # Simple progress callback that updates state directly and checks for stop
        async def progress_callback(aid, progress, message, step_data=None):
            # Check if stop was requested
            if aid in running_processes and running_processes[aid]["should_stop"]:
                print(f"🛑 Stop requested for {aid}, terminating process")
                # Set stop flag on video processor
                if hasattr(video_processor, 'should_stop'):
                    video_processor.should_stop = True
                raise Exception("Analysis stopped by user")
            
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
        
        # Clean up running process
        if analysis_id in running_processes:
            del running_processes[analysis_id]
        
        # Process next job in queue
        await process_next_queued_job()
            
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
        
        # Clean up running process
        if analysis_id in running_processes:
            del running_processes[analysis_id]

async def process_next_queued_job():
    """
    Process the next job in the queue if available
    """
    if job_queue and len(running_processes) < MAX_CONCURRENT_JOBS:
        next_job = job_queue.pop(0)
        analysis_id = next_job["analysis_id"]
        
        # Update status to processing
        if analysis_id in analysis_results:
            analysis_results[analysis_id]["status"] = "processing"
            analysis_results[analysis_id]["message"] = "Starting analysis..."
            import pytz
            singapore_tz = pytz.timezone('Asia/Singapore')
            singapore_time = datetime.now().astimezone(singapore_tz)
            analysis_results[analysis_id]["log_messages"].append({
                "timestamp": singapore_time.isoformat(),
                "message": "🚀 Starting analysis from queue",
                "progress": 5
            })
            
            # Use file path from queue
            file_path = Path(next_job.get("file_path", ""))
            
            if file_path and file_path.exists():
                # Start processing
                asyncio.create_task(process_video_with_enhanced_ai(analysis_id, file_path))
            else:
                # File not found, mark as error
                analysis_results[analysis_id]["status"] = "error"
                analysis_results[analysis_id]["message"] = "Uploaded file not found"

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
                "speech_analysis": 20,
                "body_language": 20,
                "teaching_effectiveness": 20,
                "interaction_engagement": 20,
                "presentation_skills": 20
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