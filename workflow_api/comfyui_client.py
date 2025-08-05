"""
ComfyUI API Client for Texture Generation Workflow
"""

import json
import uuid
import websocket
import urllib.request
import urllib.parse
import urllib.error
import requests
import io
import random
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
import time


class ComfyUIClient:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        
    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Queue a prompt for execution and return the prompt ID."""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        return result['prompt_id']
    
    def get_image(self, filename: str, subfolder: str, folder_type: str) -> Image.Image:
        """Download and return an image from ComfyUI server."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return Image.open(io.BytesIO(response.read()))
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get the execution history for a prompt ID."""
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for a prompt to complete execution."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout} seconds")
    
    def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI server and return the filename."""
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"http://{self.server_address}/upload/image", files=files)
            if response.status_code == 200:
                result = response.json()
                return result['name']
            else:
                raise Exception(f"Failed to upload image: {response.text}")
    
    def websocket_listen(self, prompt_id: str, callback=None):
        """Listen to websocket for real-time updates."""
        def on_message(ws, message):
            if callback:
                data = json.loads(message)
                callback(data)
        
        ws = websocket.WebSocketApp(
            f"ws://{self.server_address}/ws?clientId={self.client_id}",
            on_message=on_message
        )
        ws.run_forever()


class WorkflowManager:
    """Manages the ComfyUI workflow and parameter modifications."""
    
    def __init__(self, workflow_path: str):
        with open(workflow_path, 'r', encoding='utf-8') as f:
            self.workflow = json.load(f)
    
    def set_input_image(self, image_filename: str) -> 'WorkflowManager':
        """Set the input image filename."""
        # Node 15: LoadImage
        self.workflow["15"]["inputs"]["image"] = image_filename
        return self
    
    def set_prompt(self, prompt: str, negative_prompt: Optional[str] = None) -> 'WorkflowManager':
        """Set the text prompt for texture generation."""
        # Node 4: DiffusersIGMVSampler
        self.workflow["4"]["inputs"]["prompt"] = prompt
        if negative_prompt:
            self.workflow["4"]["inputs"]["negative_prompt"] = negative_prompt
        return self
    
    def set_image_size(self, size: int) -> 'WorkflowManager':
        """Set the image size for multiview generation."""
        # Node 18: INTConstant for image size
        self.workflow["18"]["inputs"]["value"] = size
        return self
    
    def set_mesh_input_size(self, size: int) -> 'WorkflowManager':
        """Set the mesh input image size."""
        # Node 77: INTConstant for mesh image size
        self.workflow["77"]["inputs"]["value"] = size
        return self
    
    def set_outline_size(self, size: int) -> 'WorkflowManager':
        """Set the outline thickness."""
        # Node 92: INTConstant for outline
        self.workflow["92"]["inputs"]["value"] = size
        return self
    
    def set_generation_parameters(self, steps: Optional[int] = None, 
                                 guidance_scale: Optional[float] = None,
                                 cfg_scale: Optional[float] = None) -> 'WorkflowManager':
        """Set generation parameters."""
        if steps is not None:
            # Node 4: DiffusersIGMVSampler steps
            self.workflow["4"]["inputs"]["steps"] = steps
            # Node 59: Hy3DGenerateMesh steps  
            self.workflow["59"]["inputs"]["steps"] = steps
        
        if guidance_scale is not None:
            # Node 59: Hy3DGenerateMesh guidance_scale
            self.workflow["59"]["inputs"]["guidance_scale"] = guidance_scale
        
        if cfg_scale is not None:
            # Node 4: DiffusersIGMVSampler cfg
            self.workflow["4"]["inputs"]["cfg"] = cfg_scale
        
        return self
    
    def set_seed(self, seed: Optional[int] = None) -> 'WorkflowManager':
        """Set random seeds. If None, generates random seeds."""
        if seed is None or seed == -1:
            seed = random.randint(1, 1000000)
        
        # Node 4: DiffusersIGMVSampler seed
        self.workflow["4"]["inputs"]["seed"] = seed
        # Node 59: Hy3DGenerateMesh seed
        self.workflow["59"]["inputs"]["seed"] = seed
        
        return self
    
    def set_model(self, model_name: str) -> 'WorkflowManager':
        """Set the base diffusion model."""
        # Node 9: easy ckptNames
        self.workflow["9"]["inputs"]["ckpt_name"] = model_name
        return self
    
    def set_pose_type(self, pose_type: str) -> 'WorkflowManager':
        """Set the rest pose type for rigging."""
        # Node 97: MakeItAnimatable
        self.workflow["97"]["inputs"]["rest_pose_type"] = pose_type
        return self
    
    def disable_rigging(self) -> 'WorkflowManager':
        """Disable the rigging step to speed up processing."""
        # You can remove node 97 or set it to bypass mode
        # For now, we'll just note this for implementation
        return self
    
    def get_workflow(self) -> Dict[str, Any]:
        """Get the modified workflow."""
        return self.workflow
    
    def save_workflow(self, output_path: str):
        """Save the modified workflow to a file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.workflow, f, ensure_ascii=False, indent=2)


class TextureGenerationAPI:
    """High-level API for texture generation workflow."""
    
    def __init__(self, workflow_path: str, server_address: str = "127.0.0.1:8188"):
        self.client = ComfyUIClient(server_address)
        self.workflow_path = workflow_path
    
    def generate_character(self, 
                          image_path: str,
                          prompt: str = "Highly Detailed",
                          negative_prompt: str = "watermark, ugly, deformed, noisy, blurry, low contrast",
                          image_size: int = 768,
                          mesh_input_size: int = 1024,
                          outline_size: int = 128,
                          steps: int = 10,
                          guidance_scale: float = 5.5,
                          cfg_scale: float = 2.0,
                          seed: Optional[int] = None,
                          model_name: str = "dreamshaper.safetensors",
                          pose_type: str = "大-pose",
                          enable_rigging: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a 3D character with texture from an input image.
        
        Returns:
            Tuple of (prompt_id, workflow_parameters)
        """
        
        # Upload image
        uploaded_filename = self.client.upload_image(image_path)
        
        # Prepare workflow
        workflow_manager = WorkflowManager(self.workflow_path)
        workflow_manager.set_input_image(uploaded_filename) \
                       .set_prompt(prompt, negative_prompt) \
                       .set_image_size(image_size) \
                       .set_mesh_input_size(mesh_input_size) \
                       .set_outline_size(outline_size) \
                       .set_generation_parameters(steps, guidance_scale, cfg_scale) \
                       .set_seed(seed) \
                       .set_model(model_name) \
                       .set_pose_type(pose_type)
        
        if not enable_rigging:
            workflow_manager.disable_rigging()
        
        # Queue the workflow
        workflow = workflow_manager.get_workflow()
        prompt_id = self.client.queue_prompt(workflow)
        
        parameters = {
            "image_path": image_path,
            "uploaded_filename": uploaded_filename,
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
        
        return prompt_id, parameters
    
    def get_results(self, prompt_id: str) -> Dict[str, Any]:
        """Get the results for a completed prompt."""
        history = self.client.get_history(prompt_id)
        
        if prompt_id not in history:
            return {"status": "not_found"}
        
        prompt_history = history[prompt_id]
        
        if "outputs" not in prompt_history:
            return {"status": "processing"}
        
        # Extract output files
        outputs = prompt_history["outputs"]
        results = {
            "status": "completed",
            "outputs": {},
            "files": []
        }
        
        # Process outputs from different nodes
        for node_id, node_outputs in outputs.items():
            if "images" in node_outputs:
                for image_info in node_outputs["images"]:
                    results["files"].append({
                        "type": "image",
                        "filename": image_info["filename"],
                        "subfolder": image_info.get("subfolder", ""),
                        "folder_type": image_info.get("type", "output")
                    })
            
            if "glb_path" in node_outputs:
                for path_info in node_outputs["glb_path"]:
                    results["files"].append({
                        "type": "3d_model",
                        "path": path_info
                    })
        
        return results
    
    def wait_and_get_results(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for completion and return results."""
        try:
            self.client.wait_for_completion(prompt_id, timeout)
            return self.get_results(prompt_id)
        except TimeoutError as e:
            return {"status": "timeout", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}