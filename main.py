from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import uuid
from typing import Dict, Any
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import AI processor
try:
    from ai_processor import video_processor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI processor not available. Running in mock mode.")

# Create FastAPI app
app = FastAPI(title="Discourse Analysis API", version="2.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.vercel.app",   # All Vercel domains
        "https://discourse-analysis-frontend.vercel.app"  # Your specific Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global storage for analysis results (in production, use a real database)
analysis_results = {}

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

@app.get("/")
async def root():
    return {"message": "AI-Powered Discourse Analysis API is running!", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_services": {
            "openai": "configured" if os.getenv('OPENAI_API_KEY') else "missing",
            "ai_processor": "available" if AI_AVAILABLE else "missing"
        }
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a lecture video for AI-powered analysis
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
    
    # Check file size (this is approximate, as we're streaming)
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
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
        
        # Initialize analysis status
        analysis_results[analysis_id] = {
            "status": "processing",
            "progress": 5,
            "message": "File uploaded successfully. Starting AI analysis...",
            "filename": file.filename,
            "file_size": file_path.stat().st_size
        }
        
        # Start analysis in background
        if AI_AVAILABLE:
            asyncio.create_task(process_video_with_ai(analysis_id, file_path))
        else:
            asyncio.create_task(process_video_mock(analysis_id, file_path))
        
        return {
            "analysis_id": analysis_id,
            "message": "Video uploaded successfully. AI analysis started.",
            "filename": file.filename,
            "estimated_time": "3-5 minutes" if AI_AVAILABLE else "10 seconds (mock)"
        }
        
    except Exception as e:
        # Clean up file if something went wrong
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Could not process file: {str(e)}")

@app.get("/analysis-status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """
    Check the status of an AI analysis
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

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

async def update_progress(analysis_id: str, progress: int, message: str, details: dict = None):
    """
    Helper function to update analysis progress with optional details
    """
    if analysis_id in analysis_results:
        # Preserve existing data and only update progress fields
        analysis_results[analysis_id]["progress"] = progress
        analysis_results[analysis_id]["message"] = message
        if details:
            analysis_results[analysis_id]["details"] = details
        
        print(f"DEBUG: Updated progress for {analysis_id}: {progress}% - {message}")

async def process_video_with_ai(analysis_id: str, file_path: Path):
    """
    Process video analysis using real AI services
    """
    try:
        # Define progress callback that properly updates state
        async def progress_callback(aid, progress, message):
            await update_progress(aid, progress, message)
        
        # Run the AI analysis
        results = await video_processor.process_video(
            video_path=file_path,
            analysis_id=analysis_id,
            progress_callback=progress_callback
        )
        
        # Update with final results - preserve status and add results
        if analysis_id in analysis_results:
            analysis_results[analysis_id].update({
                "status": "completed",
                "progress": 100,
                "message": "AI analysis completed successfully!",
                "results": results,
                "processing_time": "Real AI analysis complete"
            })
        
        # Clean up uploaded file after successful processing
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up file {file_path}: {e}")
            
    except Exception as e:
        print(f"Analysis failed for {analysis_id}: {str(e)}")
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

async def process_video_mock(analysis_id: str, file_path: Path):
    """
    Fallback mock processing if AI is not available
    """
    try:
        # Initialize progress tracking
        analysis_results[analysis_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Extracting audio and video frames..."
        }
        
        # Simulate processing steps
        await asyncio.sleep(2)
        analysis_results[analysis_id]["progress"] = 30
        analysis_results[analysis_id]["message"] = "Analyzing speech patterns..."
        
        await asyncio.sleep(3)
        analysis_results[analysis_id]["progress"] = 60
        analysis_results[analysis_id]["message"] = "Analyzing gestures and body language..."
        
        await asyncio.sleep(3)
        analysis_results[analysis_id]["progress"] = 80
        analysis_results[analysis_id]["message"] = "Generating final report..."
        
        await asyncio.sleep(2)
        
        # Create mock analysis results
        final_results = create_mock_analysis_results()
        
        analysis_results[analysis_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Analysis completed successfully! (Mock mode)",
            "results": final_results
        }
        
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()
            
    except Exception as e:
        analysis_results[analysis_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Analysis failed: {str(e)}"
        }

def create_mock_analysis_results() -> Dict[str, Any]:
    """
    Create mock analysis results (fallback when AI is not available)
    """
    return {
        "overall_score": 7.8,
        "speech_analysis": {
            "score": 8.2,
            "clarity": 8.5,
            "pace": 7.8,
            "filler_words": 8.0,
            "enthusiasm": 8.5,
            "feedback": [
                "Excellent clarity in speech delivery",
                "Good variation in pace to maintain engagement",
                "Consider reducing use of filler words slightly",
                "Great enthusiasm and energy throughout"
            ]
        },
        "body_language": {
            "score": 7.5,
            "eye_contact": 7.8,
            "gestures": 7.2,
            "posture": 8.0,
            "movement": 7.0,
            "feedback": [
                "Good eye contact with camera/audience",
                "Natural hand gestures support the content",
                "Confident posture throughout presentation",
                "Consider moving around more to engage different areas"
            ]
        },
        "teaching_effectiveness": {
            "score": 7.6,
            "content_flow": 8.0,
            "explanations": 7.5,
            "examples": 7.2,
            "engagement": 7.8,
            "feedback": [
                "Logical flow of information",
                "Clear explanations of complex concepts",
                "Good use of examples to illustrate points",
                "Strong techniques to maintain student engagement"
            ]
        },
        "presentation_skills": {
            "score": 8.0,
            "professionalism": 8.5,
            "energy": 8.2,
            "time_management": 7.5,
            "conclusion": 7.8,
            "feedback": [
                "Very professional presentation style",
                "High energy level maintains interest",
                "Generally good time management",
                "Strong conclusion that summarizes key points"
            ]
        },
        "improvement_suggestions": [
            "Practice reducing filler words during pauses",
            "Incorporate more movement around the teaching space",
            "Add more concrete examples to complex explanations",
            "Consider interactive elements to boost engagement",
            "Work on pacing to ensure all content fits time slot"
        ],
        "strengths": [
            "Excellent speaking clarity and voice projection",
            "Professional demeanor and confident presence",
            "Good use of gestures to emphasize points",
            "Strong content organization and flow",
            "High energy and enthusiasm for the subject"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)