# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Make-It-Animatable is a CVPR 2025 research project for creating animation-ready 3D characters from static models. It uses deep learning to predict skeletal structures, bone weights, and pose transformations, enabling automatic rigging and animation of 3D characters using Mixamo animations.

## Core Architecture

The system operates through a multi-stage pipeline:

1. **Point Cloud Processing**: Characters are represented as point clouds (32,768 points by default)
2. **Multi-Task Learning**: Three separate models predict:
   - `joints`: Joint positions and skeletal structure 
   - `bw`: Blend weights for skinning
   - `pose`: Pose transformations (rotation + translation)
3. **Blender Integration**: Real-time character-animation pairing using Blender's `bpy` API
4. **3D Gaussian Splatting**: Optional rendering using integrated 3DGS addon

### Key Components

- **`model.py`**: Core PCAE (Point Cloud AutoEncoder) architecture with transformer-based attention
- **`engine.py`**: Training engine with GT data structures and loss computation  
- **`util/dataset_mixamo.py`**: Mixamo dataset handling with 65-joint skeletal structure
- **`util/blender_utils.py`**: Blender integration for FBX processing and animation application
- **`app.py`**: Gradio web interface for inference

## Development Commands

### Environment Setup
```bash
conda create -n mia python=3.11
conda activate mia
pip install -r requirements.txt
```

### Training
```bash
# Full training pipeline (requires multi-GPU)
bash train.sh

# Individual model training
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) train.py [args]
```

### Inference
```bash
# Web interface
python app.py

# Blender-based processing
python app_blender.py
```

### Testing
```bash
# Test core functionality without Blender
python test_core.py

# Test imports and dependencies
python test_import.py
```

## Data Pipeline

### Mixamo Integration
- Characters and animations are paired online during training
- 65-joint skeletal structure with hierarchical relationships defined in `KINEMATIC_TREE`
- Bone indices mapped via `BONES_IDX_DICT` for consistent joint ordering
- Supports both standard Mixamo and custom character sets (3DBiCar dataset)

### Blender Processing
The system heavily relies on Blender for:
- FBX import/export and format conversion
- Character-animation retargeting
- Mesh sampling and ground truth extraction
- Real-time pose application and validation

## Model Training Strategy

Models are trained separately for efficiency:
1. **Coarse joints model**: Initial joint localization (20 epochs)
2. **Blend weights model**: Skinning weight prediction  
3. **Fine joints model**: Refined joint positions with causal attention
4. **Pose model**: Transformation prediction using ortho6d representation

Each model uses different hyperparameters optimized for A100 80G GPUs. Reduce `--batch_size` and `--sample_frames` for lower VRAM.

## Key Dependencies

- **PyTorch 2.1.2** with CUDA 12.1 support
- **Blender 4.3.0** (bpy module) for 3D processing
- **PyTorch3D** for 3D transformations and rendering
- **torch_cluster** for point cloud operations (FPS sampling)
- **trimesh** for mesh processing
- **Gradio 5.25+** for web interface

## Development Notes

### Coordinate Systems
- Uses right-handed coordinate system consistent with Blender
- Transformations handled via PyTorch3D's Transform3d class
- Pose representations use ortho6d for stability (not dual quaternions as in paper)

### Memory Management  
- Point clouds sampled to 32,768 points for memory efficiency
- Online data pairing reduces storage requirements
- Gradient accumulation recommended for large batch training

### Blender Integration Caveats
- Requires full Blender installation, not just bpy-stub
- Animation retargeting may need manual adjustment for non-humanoid characters
- FBX format compatibility issues with some modeling software

### ComfyUI Integration
- **API-Based Architecture**: Uses isolated FastAPI server to avoid library conflicts with ComfyUI
- **Environment Separation**: Independent `.venv` in `api_server/` directory with specific dependency versions
- **Automated Server Management**: API server starts automatically when ComfyUI nodes are used
- **Node Implementation**: `comfyui_node.py` provides `MakeItAnimatable` and `MakeItAnimatableLoader` nodes
- **Setup Process**: Run `python setup_api.py` to create isolated environment and install dependencies

### ComfyUI Development Commands
```bash
# Manual API server start
source .venv/bin/activate && cd api_server && python main.py
```

## Current Status (2025-08-04)

### Working Components ✅
- **Model Inference Pipeline**: All 5 models (bw, bw_normal, joints, joints_coarse, pose) load and run successfully
- **FastAPI Server**: Isolated environment with proper dependency versions 
- **Pose Data Conversion**: Fixed `to_pose_matrix` function to support `"ortho6d"` mode, converting 6D representation to 4x4 matrices
- **Data Format Handling**: Proper conversion from torch tensors to numpy arrays with correct dimensions
- **Basic Visualization**: Generates `rest_lbs.glb`, `bw.glb`, `joints.glb` outputs in `api_server/test/`

### Current Issues ❌
- **Blender Integration**: `app_blender.py` crashes with Segmentation fault (exit code 35584)
- **FBX Export**: Cannot generate rigged FBX files due to Blender crash
- **Animation Integration**: Blender rigging process fails, falls back to mesh-only export

### Technical Details
- **API Endpoint**: `POST /animate` at `http://127.0.0.1:8765`
- **Output Format**: Currently GLB format (Blender rigging disabled due to segfault)
- **Processing Time**: ~17 seconds for full pipeline
- **Data Flow**: Input GLB → Point Cloud → Model Inference → Pose Conversion → Visualization → Output GLB

### Next Steps for New Session
1. **Investigate Blender Segfault**: Debug `app_blender.py` crash issue
2. **Alternative Rigging**: Consider using different 3D processing library instead of Blender bpy
3. **FBX Export Fix**: Resolve rigged model export to enable proper animation
4. **ComfyUI Node Testing**: Verify end-to-end workflow from ComfyUI interface

### Key Files Modified
- `api_server/main.py`: Added pose matrix conversion logic
- `util/utils.py`: Added `"ortho6d"` mode support to `to_pose_matrix` function
- `comfyui_node.py`: Configured for GLB output format