#!/usr/bin/env python3
"""
ComfyUI Node for Make-It-Animatable
Communicates with isolated API server to avoid library conflicts
"""

import os
import json
import tempfile
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any, Tuple

import folder_paths

class MakeItAnimatableNode:
    """
    ComfyUI Node for Make-It-Animatable character animation
    """
    
    def __init__(self):
        self.api_url = "http://127.0.0.1:8765"
        self.server_process = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_3d_model": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "animation_file": ("STRING", {"default": "", "multiline": False}),
                
                # Input Settings
                "is_gs": ("BOOLEAN", {"default": False}),
                "opacity_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "no_fingers": ("BOOLEAN", {"default": True}),
                "rest_pose_type": (["No", "T-pose", "A-pose", "大-pose"], {"default": "No"}),
                "rest_parts": ("STRING", {"default": "[]", "multiline": False}), # JSON array
                
                # Weight Settings  
                "input_normal": ("BOOLEAN", {"default": False}),
                "bw_fix": ("BOOLEAN", {"default": True}),
                "bw_vis_bone": ("STRING", {"default": "LeftArm"}),
                
                # Animation Settings
                "reset_to_rest": ("BOOLEAN", {"default": True}),
                "retarget": ("BOOLEAN", {"default": True}),
                "inplace": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)  # Returns path to animated model
    RETURN_NAMES = ("animated_model_path",)
    FUNCTION = "animate_character"
    CATEGORY = "3D/Animation"
    DESCRIPTION = "Convert 3D character models to animation-ready format using Make-It-Animatable"

    def ensure_api_server(self) -> bool:
        """
        Ensure API server is running, start if necessary
        """
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        
        # Start server if not running
        print("🚀 Starting Make-It-Animatable API server...")
        return self.start_api_server()
    
    def start_api_server(self) -> bool:
        """
        Start the isolated API server
        """
        try:
            # Path to the API server
            api_server_dir = Path(__file__).parent / "api_server"
            
            # Check if virtual environment exists
            venv_path = api_server_dir / ".venv"
            if not venv_path.exists():
                print("❌ API server virtual environment not found. Run setup first.")
                return False
            
            # Start server in background
            python_path = venv_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            server_script = api_server_dir / "main.py"
            
            self.server_process = subprocess.Popen([
                str(python_path), str(server_script)
            ], cwd=str(api_server_dir))
            
            # Wait for server to start
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get(f"{self.api_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ API server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("❌ Failed to start API server")
            return False
            
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def animate_character(
        self,
        input_3d_model: str,
        output_path: str = "",
        animation_file: str = "",
        is_gs: bool = False,
        opacity_threshold: float = 0.01,
        no_fingers: bool = True,
        rest_pose_type: str = "No",
        rest_parts: str = "[]",
        input_normal: bool = False,
        bw_fix: bool = True,
        bw_vis_bone: str = "LeftArm",
        reset_to_rest: bool = True,
        retarget: bool = True,
        inplace: bool = True,
    ) -> Tuple[str]:
        """
        Main processing function
        """
        
        # Validate inputs
        if not input_3d_model or not os.path.exists(input_3d_model):
            raise ValueError(f"Input 3D model file not found: {input_3d_model}")
        
        # Ensure API server is running
        if not self.ensure_api_server():
            raise RuntimeError("Failed to start Make-It-Animatable API server")
        
        try:
            # Parse rest_parts JSON
            try:
                rest_parts_list = json.loads(rest_parts) if rest_parts else []
            except json.JSONDecodeError:
                rest_parts_list = []
            
            # Prepare request data
            request_data = {
                "is_gs": is_gs,
                "opacity_threshold": opacity_threshold,
                "no_fingers": no_fingers,
                "rest_pose_type": rest_pose_type,
                "rest_parts": rest_parts_list,
                "input_normal": input_normal,
                "bw_fix": bw_fix,
                "bw_vis_bone": bw_vis_bone,
                "reset_to_rest": reset_to_rest,
                "retarget": retarget,
                "inplace": inplace,
            }
            
            # Prepare files
            files = {}
            with open(input_3d_model, 'rb') as f:
                files['input_file'] = (os.path.basename(input_3d_model), f, 'application/octet-stream')
                
                if animation_file and os.path.exists(animation_file):
                    with open(animation_file, 'rb') as af:
                        files['animation_file'] = (os.path.basename(animation_file), af, 'application/octet-stream')
                
                # Send request to API
                print("🎯 Processing character animation...")
                response = requests.post(
                    f"{self.api_url}/animate",
                    files=files,
                    data={"request": json.dumps(request_data)},
                    timeout=300  # 5 minute timeout
                )
            
            if response.status_code != 200:
                raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
            
            # Determine output path
            if output_path:
                # Use user-specified output path
                final_output_path = Path(output_path)
                # Create directory if it doesn't exist
                final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Use default ComfyUI/output/3D directory
                base_name = Path(input_3d_model).stem
                comfyui_root = Path(folder_paths.base_path)  # ComfyUI root directory
                output_dir = comfyui_root / "output" / "3D"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Use .fbx extension for rigged models
                final_output_path = output_dir / f"{base_name}_rigged.fbx"
            
            with open(final_output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Animation complete! Output: {final_output_path}")
            return (str(final_output_path),)
            
        except Exception as e:
            raise RuntimeError(f"Animation processing failed: {str(e)}")


class MakeItAnimatableLoader:
    """
    Utility node to load 3D models for Make-It-Animatable
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "load_model"
    CATEGORY = "3D/Animation"
    
    def load_model(self, model_path: str) -> Tuple[str]:
        """Load and validate 3D model file"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Validate file extension
        valid_extensions = ['.glb', '.fbx', '.ply', '.obj', '.dae']
        if not any(model_path.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"Unsupported file format. Supported: {valid_extensions}")
        
        return (model_path,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MakeItAnimatable": MakeItAnimatableNode,
    "MakeItAnimatableLoader": MakeItAnimatableLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MakeItAnimatable": "Make-It-Animatable",
    "MakeItAnimatableLoader": "3D Model Loader (MIA)",
}