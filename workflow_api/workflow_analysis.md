# Workflow Analysis: main_workflow.json

## Overview

This workflow performs a complete 3D character generation and texturing pipeline:
1. Image input and preprocessing
2. 3D mesh generation using Hunyuan3D
3. Multi-view texture generation using IG2MV
4. Texture baking and inpainting
5. Character rigging and animation

## Key Nodes and Dynamic Parameters

### Input Parameters
- **Node 15 (LoadImage)**: Input image file
  - `inputs.image`: Image filename to process
  
### Text Generation Parameters  
- **Node 4 (DiffusersIGMVSampler)**: Multi-view texture generation
  - `inputs.prompt`: Text prompt for texture generation ("Highly Detailed")
  - `inputs.negative_prompt`: Negative prompt
  - `inputs.steps`: Generation steps (10)
  - `inputs.cfg`: CFG scale (2)
  - `inputs.seed`: Random seed (2)

### Size Parameters
- **Node 18 (INTConstant)**: Image size for multiview
  - `outputs[0]`: Used for texture generation size (768)
  
- **Node 77 (INTConstant)**: Mesh input image size  
  - `outputs[0]`: Input image processing size (1024)
  
- **Node 92 (INTConstant)**: Outline thickness
  - `outputs[0]`: Border outline size (128)

### 3D Generation Parameters
- **Node 59 (Hy3DGenerateMesh)**: 3D mesh generation
  - `inputs.guidance_scale`: Generation guidance (5.5)
  - `inputs.steps`: Generation steps (50)
  - `inputs.seed`: Random seed (42)

### Model Selection
- **Node 9 (easy ckptNames)**: Base model selection
  - `inputs.ckpt_name`: Checkpoint model ("dreamshaper.safetensors")

- **Node 61 (Hy3DModelLoader)**: 3D model selection
  - `inputs.model`: 3D generation model ("hunyuan3d-dit-v2-0-fp16.safetensors")

### Animation Parameters
- **Node 97 (MakeItAnimatable)**: Character rigging
  - `inputs.rest_pose_type`: Rest pose type ("大-pose")
  - `inputs.opacity_threshold`: Opacity threshold (0.01)
  - `inputs.no_fingers`: Finger handling (true)

## Output Nodes

- **Node 30**: Preview of vertex inpainted texture
- **Node 32**: Preview of fully inpainted texture  
- **Node 33**: Preview of initial baked texture
- **Node 83**: Preview of generated 3D mesh
- **Node 43**: Preview of textured 3D model
- **Node 98**: Preview of final rigged character

## File Outputs

- **Node 64**: Raw 3D mesh export (GLB format)
- **Node 40**: Textured 3D model export (GLB format)  
- **Node 97**: Final rigged character export (FBX format)

## API-Friendly Parameters

For API implementation, these parameters should be configurable:

1. **input_image**: Upload file or file path
2. **prompt**: Text description for texture generation
3. **negative_prompt**: What to avoid in generation
4. **image_size**: Size for multiview generation (768, 1024, etc.)
5. **mesh_input_size**: Input image processing size  
6. **steps**: Generation quality vs speed tradeoff
7. **guidance_scale**: How closely to follow the prompt
8. **seed**: For reproducible results
9. **model_name**: Base diffusion model to use
10. **pose_type**: Character rest pose for rigging