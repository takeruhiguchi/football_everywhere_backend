# Commit Log

## 2025/08/06 40ddd63: Fix workflow API file detection and response optimization
- Fix ComfyUI results extraction in comfyui_client.py to properly handle result arrays
- Add workaround in api_server.py to extract files from results when job.files is empty  
- Modify API to return only rigged character FBX files instead of all generated files
- Simplify job status response to remove heavy results field and return minimal info
- Add caching for processed file information to improve performance
- Update test_complete_api.py with FBX-only filtering and result extraction workaround

## 2025/08/05 7fee26e: Fix ComfyUI workflow parameter setting
- Update WorkflowManager to modify widgets_values instead of inputs
- Fix LoadImage node to properly set image filename in widgets_values
- Correct parameter indices for DiffusersIGMVSampler and Hy3DGenerateMesh nodes
- Update prompt, seed, steps, guidance_scale, and cfg_scale setting methods
- Fix model selection and pose type configuration
- Resolve '15' error by properly updating workflow node values

## 2025/08/05 74b41f4: Implement ComfyUI Texture Generation API
- Add comprehensive API implementation for main_workflow.json
- Create ComfyUI client for workflow execution and parameter modification  
- Implement FastAPI server with REST endpoints for character generation
- Add job status monitoring and progress tracking
- Include complete documentation and usage examples
- Support dynamic parameter configuration (prompt, image size, steps, etc.)
- Add file upload handling and validation
- Implement background job processing with status updates