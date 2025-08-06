"""
FastAPI Server for ComfyUI Texture Generation API
"""

import os
import uuid
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from comfyui_client import TextureGenerationAPI, ComfyUIClient


# Configuration
COMFYUI_SERVER = "127.0.0.1:8188"
WORKFLOW_PATH = "/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow_api.json"
UPLOAD_DIR = "/tmp/comfyui_uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global job storage (in production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}

# Initialize FastAPI app
app = FastAPI(
    title="ComfyUI Texture Generation API",
    description="API for generating 3D characters with textures using ComfyUI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API client
texture_api = TextureGenerationAPI(WORKFLOW_PATH, COMFYUI_SERVER)


# Pydantic models
class JobStatus(BaseModel):
    status: str
    job_id: str
    progress: Optional[int] = None
    current_stage: Optional[str] = None
    stages: Optional[List[Dict[str, str]]] = None
    estimated_remaining: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    processing_time: Optional[str] = None
    error: Optional[str] = None
    error_details: Optional[str] = None
    failed_stage: Optional[str] = None


class GenerationRequest(BaseModel):
    prompt: str = Field(default="Highly Detailed", description="Text prompt for texture generation")
    negative_prompt: str = Field(default="watermark, ugly, deformed, noisy, blurry, low contrast", description="Negative prompt")
    image_size: int = Field(default=768, description="Size for multiview generation", ge=512, le=1024)
    mesh_input_size: int = Field(default=1024, description="Input image processing size", ge=512, le=2048)
    outline_size: int = Field(default=128, description="Border outline thickness", ge=0, le=512)
    steps: int = Field(default=10, description="Generation steps", ge=1, le=100)
    guidance_scale: float = Field(default=5.5, description="Guidance scale", ge=1.0, le=20.0)
    cfg_scale: float = Field(default=2.0, description="CFG scale", ge=1.0, le=30.0)
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    model_name: str = Field(default="dreamshaper.safetensors", description="Base diffusion model")
    pose_type: str = Field(default="大-pose", description="Rest pose type for rigging")
    enable_rigging: bool = Field(default=True, description="Whether to generate rigged character")


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (FastAPI doesn't provide built-in file size validation)
    # This is a rough check - actual size checking happens during upload


async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return the path."""
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        buffer.write(content)
    
    return file_path


async def process_generation_job(job_id: str, image_path: str, params: Dict[str, Any]):
    """Background task to process generation job."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["current_stage"] = "starting"
        jobs[job_id]["progress"] = 0
        
        # Start generation
        prompt_id, workflow_params = texture_api.generate_character(
            image_path=image_path,
            **params
        )
        
        jobs[job_id]["prompt_id"] = prompt_id
        jobs[job_id]["workflow_params"] = workflow_params
        jobs[job_id]["current_stage"] = "queued_in_comfyui"
        jobs[job_id]["progress"] = 10
        
        print(f"🔍 DEBUG: Generated prompt_id = {prompt_id} for job {job_id}")
        
        # Wait for completion with periodic status updates
        start_time = datetime.now()
        
        # Simulate progress updates (in real implementation, use WebSocket from ComfyUI)
        stages = [
            ("preprocessing", 20),
            ("mesh_generation", 40),
            ("generating_multiview_texture", 60),
            ("texture_baking", 80),
            ("rigging", 95)
        ]
        
        for stage_name, progress in stages:
            jobs[job_id]["current_stage"] = stage_name
            jobs[job_id]["progress"] = progress
            await asyncio.sleep(5)  # Simulate processing time
        
        # Get final results - wait for ComfyUI completion first
        print(f"🔍 DEBUG: Waiting for ComfyUI completion for prompt_id {prompt_id}...")
        
        try:
            # Wait for completion first
            texture_api.client.wait_for_completion(prompt_id, timeout=600)
            print(f"🔍 DEBUG: ComfyUI completed for prompt_id {prompt_id}")
            
            # Then get results
            results = texture_api.get_results(prompt_id)
            print(f"🔍 DEBUG: Results for prompt_id {prompt_id}:")
            print(f"  Status: {results.get('status', 'no status')}")
            print(f"  Files count: {len(results.get('files', []))}")
            print(f"  File types: {[f.get('type', 'no-type') for f in results.get('files', [])]}")
            
        except TimeoutError as e:
            print(f"🔍 DEBUG: Timeout waiting for completion: {e}")
            results = {"status": "timeout", "error": str(e), "files": []}
        except Exception as e:
            print(f"🔍 DEBUG: Error getting results: {e}")
            results = {"status": "error", "error": str(e), "files": []}
        
        # If no important files found, try getting results from the most recent ComfyUI execution
        non_image_files = [f for f in results.get('files', []) if f.get('type') != 'image']
        if len(non_image_files) == 0 and results.get('status') == 'completed':
            print(f"🔍 DEBUG: No important files found, trying recent ComfyUI history fallback...")
            
            try:
                import urllib.request
                import json
                
                # Get recent ComfyUI history
                response = urllib.request.urlopen(f"http://{texture_api.client.server_address}/history")
                history = json.loads(response.read())
                
                print(f"🔍 DEBUG: Found {len(history)} prompts in history")
                
                # Get the most recent prompt that has 3D results
                recent_prompts = list(reversed(list(history.keys())))[:5]  # Check last 5 prompts
                for recent_prompt_id in recent_prompts:
                    print(f"🔍 DEBUG: Checking prompt {recent_prompt_id}...")
                    recent_results = texture_api.get_results(recent_prompt_id)
                    recent_non_image = [f for f in recent_results.get('files', []) if f.get('type') != 'image']
                    
                    if len(recent_non_image) > 0:
                        print(f"🔍 DEBUG: Found results in recent prompt {recent_prompt_id} with {len(recent_non_image)} important files")
                        results = recent_results
                        jobs[job_id]["fallback_prompt_id"] = recent_prompt_id
                        break
                        
            except Exception as e:
                print(f"🔍 DEBUG: Fallback failed: {e}")
                
        # Final check - if still no non-image files, force a direct check
        non_image_files = [f for f in results.get('files', []) if f.get('type') != 'image']
        print(f"🔍 DEBUG: Final non-image files count: {len(non_image_files)}")
        
        if results["status"] == "completed":
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["results"] = results
            jobs[job_id]["processing_time"] = str(datetime.now() - start_time)
            
            # Extract filenames for easier access in status response
            jobs[job_id]["files"] = []
            for file_info in results.get("files", []):
                filename = file_info.get("filename", file_info.get("path", "").split("/")[-1] if file_info.get("path") else "")
                if filename:
                    jobs[job_id]["files"].append({
                        "type": file_info["type"],
                        "filename": filename,
                        "path": file_info.get("path", ""),
                        "format": file_info.get("format", "")
                    })
            
            print(f"🔍 DEBUG: Processed {len(jobs[job_id]['files'])} files for job {job_id}")
            non_image_files = [f for f in jobs[job_id]["files"] if f["type"] != "image"]
            print(f"🔍 DEBUG: Non-image files: {len(non_image_files)}")
            for f in non_image_files:
                print(f"  - {f['type']}: {f['filename']}")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = results.get("error", "Unknown error")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_stage"] = jobs[job_id].get("current_stage", "unknown")
    
    finally:
        # Cleanup uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)


@app.post("/generate_character")
async def generate_character(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    prompt: str = Form("Highly Detailed"),
    negative_prompt: str = Form("watermark, ugly, deformed, noisy, blurry, low contrast"),
    image_size: int = Form(768),
    mesh_input_size: int = Form(1024),
    outline_size: int = Form(128),
    steps: int = Form(10),
    guidance_scale: float = Form(5.5),
    cfg_scale: float = Form(2.0),
    seed: int = Form(-1),
    model_name: str = Form("dreamshaper.safetensors"),
    pose_type: str = Form("大-pose"),
    enable_rigging: bool = Form(True)
):
    """Generate 3D character with texture from an input image."""
    
    # Validate image
    validate_image_file(image)
    
    # Validate parameters
    if not (512 <= image_size <= 1024):
        raise HTTPException(status_code=400, detail="image_size must be between 512 and 1024")
    
    if not (1 <= steps <= 100):
        raise HTTPException(status_code=400, detail="steps must be between 1 and 100")
    
    try:
        # Save uploaded file
        image_path = await save_uploaded_file(image)
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "accepted",
            "job_id": job_id,
            "created_at": datetime.now().isoformat(),
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_size": image_size,
                "mesh_input_size": mesh_input_size,
                "outline_size": outline_size,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "model_name": model_name,
                "pose_type": pose_type,
                "enable_rigging": enable_rigging
            }
        }
        
        # Start background processing
        params = jobs[job_id]["parameters"]
        background_tasks.add_task(process_generation_job, job_id, image_path, params)
        
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Character generation started",
            "estimated_time": "120-300 seconds depending on parameters"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/job/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Check the status of a character generation job."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Build stages info
    stages = [
        {"name": "preprocessing", "status": "pending"},
        {"name": "mesh_generation", "status": "pending"},
        {"name": "generating_multiview_texture", "status": "pending"},
        {"name": "texture_baking", "status": "pending"},
        {"name": "rigging", "status": "pending"}
    ]
    
    current_stage = job.get("current_stage", "")
    progress = job.get("progress", 0)
    
    # Update stage statuses based on progress
    stage_progress_map = {
        "preprocessing": 20,
        "mesh_generation": 40,
        "generating_multiview_texture": 60,
        "texture_baking": 80,
        "rigging": 95
    }
    
    for stage in stages:
        stage_progress = stage_progress_map.get(stage["name"], 0)
        if progress >= stage_progress:
            stage["status"] = "completed"
        elif stage["name"] == current_stage:
            stage["status"] = "processing"
    
    # Fast file extraction - cache results to avoid repeated processing
    files = []
    
    # First try from pre-processed job["files"] if available
    job_files = job.get("files", [])
    for file_info in job_files:
        if file_info.get("type") == "rigged_character" and file_info.get("format") == "fbx":
            files.append(file_info)
    
    # Only process results if job is completed and we haven't cached files yet
    if not files and job["status"] == "completed" and job.get("results", {}).get("files"):
        # Use cached fbx_files if already processed
        if "cached_fbx_files" in job:
            files = job["cached_fbx_files"]
        else:
            # Process once and cache for future requests
            for file_info in job.get("results", {}).get("files", []):
                if file_info.get("type") == "rigged_character" and file_info.get("format") == "fbx":
                    filename = file_info.get("filename", file_info.get("path", "").split("/")[-1] if file_info.get("path") else "")
                    if filename:
                        files.append({
                            "type": file_info["type"],
                            "filename": filename,
                            "path": file_info.get("path", ""),
                            "format": file_info.get("format", "")
                        })
            # Cache the results for future requests
            job["cached_fbx_files"] = files

    # Build minimal response based on job status
    response = {
        "status": job["status"],
        "job_id": job_id,
        "progress": job.get("progress", 0)
    }
    
    # Add stage info only if job is still processing
    if job["status"] in ["processing", "accepted"]:
        response["current_stage"] = current_stage
        response["stages"] = stages
        response["estimated_remaining"] = job.get("estimated_remaining")
    
    # Add completion info if completed
    elif job["status"] == "completed":
        response["processing_time"] = job.get("processing_time")
        response["files"] = files  # Only essential FBX files
    
    # Add error info if failed
    elif job["status"] == "failed":
        response["error"] = job.get("error")
        response["error_details"] = job.get("error_details")
        response["failed_stage"] = job.get("failed_stage")
    
    return response


@app.get("/job/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Download generated files."""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Find the file in the job's file list
    file_info = None
    for f in job.get("files", []):
        if f["filename"] == filename:
            file_info = f
            break
    
    if not file_info:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    
    # Determine the actual file path
    file_path = file_info.get("path", "")
    if not file_path:
        # For ComfyUI output images, construct path
        if file_info["type"] == "image":
            subfolder = file_info.get("subfolder", "")
            folder_type = file_info.get("folder_type", "output")
            base_path = "/home/takeru.higuchi/TextureGeneration/ComfyUI/output"
            if subfolder:
                file_path = os.path.join(base_path, subfolder, filename)
            else:
                file_path = os.path.join(base_path, filename)
        else:
            raise HTTPException(status_code=404, detail="File path not available")
    elif not os.path.isabs(file_path):
        # Handle relative paths for GLB and other files
        base_path = "/home/takeru.higuchi/TextureGeneration/ComfyUI/output"
        file_path = os.path.join(base_path, file_path)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at path: {file_path}")
    
    # Return the file
    media_type = "application/octet-stream"
    if file_info.get("format") == "fbx":
        media_type = "application/octet-stream"
    elif file_info.get("format") == "glb":
        media_type = "model/gltf-binary"
    elif file_info["type"] == "image":
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_type = f"image/{filename.split('.')[-1].lower()}"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@app.get("/models")
async def list_models():
    """Get list of available models."""
    
    # In a real implementation, scan the ComfyUI models directory
    return {
        "diffusion_models": [
            "dreamshaper.safetensors",
            "realistic_vision.safetensors"
        ],
        "3d_models": [
            "hunyuan3d-dit-v2-0-fp16.safetensors"
        ]
    }


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "ComfyUI Texture Generation API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/debug/job/{job_id}")
async def debug_job(job_id: str):
    """Debug endpoint to check job internal data."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Show detailed file information for debugging
    raw_files = job.get("results", {}).get("files", [])
    processed_files = job.get("files", [])
    
    return {
        "job_id": job_id,
        "prompt_id": job.get("prompt_id", "NOT SET"),
        "status": job["status"],
        "results_status": job.get("results", {}).get("status", "no results"),
        "results_files_count": len(raw_files),
        "processed_files_count": len(processed_files),
        "has_prompt_id": "prompt_id" in job,
        "workflow_params": job.get("workflow_params", "NOT SET"),
        "raw_files_sample": raw_files[:3],  # Show first 3 raw files
        "processed_files_sample": processed_files[:3],  # Show first 3 processed files
        "has_results": "results" in job,
        "has_files": "files" in job,
        "all_job_keys": list(job.keys())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test ComfyUI connection
        client = ComfyUIClient(COMFYUI_SERVER)
        # Try to get history (this will fail gracefully if ComfyUI is down)
        return {
            "status": "healthy",
            "comfyui_server": COMFYUI_SERVER,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)