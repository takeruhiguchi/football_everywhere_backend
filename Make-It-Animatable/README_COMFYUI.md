# Make-It-Animatable for ComfyUI

ComfyUI custom node implementation of Make-It-Animatable with isolated API architecture to avoid library conflicts.

## 🏗️ Architecture

```
ComfyUI Environment
├── ComfyUI Nodes (comfyui_node.py)
│   └── HTTP Client
└── API Server (Isolated .venv)
    ├── FastAPI Server
    ├── Make-It-Animatable Core
    └── Independent Dependencies
```

## 🚀 Installation

### 1. Initial Setup

```bash
cd /path/to/ComfyUI/custom_nodes/Make-It-Animatable
python setup_api.py
```

This will:
- Create isolated Python environment in `api_server/.venv/`
- Install all required dependencies without affecting ComfyUI
- Copy necessary Make-It-Animatable files
- Set up the API server

### 2. Download Required Data

```bash
# Download pretrained models (if not already done)
cd api_server
git -C /tmp/hf-data lfs pull -I output/best/new
mkdir -p output/best && cp -r /tmp/hf-data/output/best/new output/best/

# Download example data and Mixamo animations
git -C /tmp/hf-data lfs pull -I data
cp -r /tmp/hf-data/data/* data/

# Download FBX converter
wget https://github.com/facebookincubator/FBX2glTF/releases/download/v0.9.7/FBX2glTF-linux-x64 -O util/FBX2glTF
chmod +x util/FBX2glTF
```

## 🎮 Usage in ComfyUI

### Nodes Available

1. **3D Model Loader (MIA)**: Load and validate 3D model files
2. **Make-It-Animatable**: Main processing node

### Basic Workflow

```
3D Model Loader → Make-It-Animatable → [Output]
      ↓                    ↑
   model.glb         animation.fbx
```

### Node Parameters

#### Input Settings
- **Input 3D Model**: Path to 3D model file (`.glb`, `.fbx`, `.ply`, etc.)
- **Animation File**: Optional animation file (`.fbx`)
- **Is GS**: Whether input is Gaussian Splats
- **Opacity Threshold**: For Gaussian Splats (0.0-1.0)
- **No Fingers**: Disable finger animation (recommended for simple models)
- **Rest Pose Type**: Pre-existing pose type (`No`, `T-pose`, `A-pose`, `大-pose`)
- **Rest Parts**: JSON array of parts already in T-pose

#### Weight Settings
- **Input Normal**: Use surface normals (mesh only)
- **BW Fix**: Apply weight post-processing
- **BW Vis Bone**: Bone for weight visualization

#### Animation Settings
- **Reset to Rest**: Apply predicted T-pose
- **Retarget**: Better animation quality
- **In Place**: Keep looping animations stationary

## 🔧 API Server Details

### Endpoints

- `GET /health`: Server health check
- `POST /animate`: Main animation processing endpoint

### Server Management

The API server starts automatically when the ComfyUI node is used. Manual control:

```bash
# Start server manually
cd api_server
python start_server.py

# Or directly
.venv/bin/python main.py  # Linux/Mac
.venv\Scripts\python.exe main.py  # Windows
```

Server runs on `http://127.0.0.1:8765`

## 🐛 Troubleshooting

### Common Issues

1. **"API server failed to start"**
   - Run `python setup_api.py` again
   - Check if port 8765 is available
   - Verify virtual environment: `api_server/.venv/`

2. **"Models not initialized"**
   - Ensure pretrained models are downloaded to `api_server/output/best/new/`
   - Check server logs for CUDA/memory issues

3. **"Processing failed"**
   - Verify input file format is supported
   - Check available disk space for temporary files
   - Monitor server logs in terminal

4. **Library conflicts**
   - The API server runs in isolated environment
   - ComfyUI environment should not be affected
   - If issues persist, reinstall: `rm -rf api_server/.venv && python setup_api.py`

### Performance Tips

- **GPU Memory**: Adjust batch sizes in model if OOM errors occur
- **Processing Time**: Large models take 2-5 minutes depending on settings
- **File Sizes**: Compressed output files >50MB automatically

## 🔄 Development

### Extending Functionality

To add new features:

1. **API Side**: Modify `api_server/main.py`
2. **ComfyUI Side**: Update `comfyui_node.py`
3. **Both**: Update parameter definitions

### Testing

```bash
# Test API server
cd api_server
.venv/bin/python -c "import requests; print(requests.get('http://127.0.0.1:8765/health').json())"

# Test in ComfyUI
# Use the example workflow with test models
```

## 📝 Notes

- **Blender Integration**: Requires full Blender installation via bpy module
- **File Formats**: 
  - Input: `.glb`, `.fbx`, `.ply`, `.obj`, `.dae`
  - Output: `.fbx` (meshes), `.blend` (Gaussian Splats)
- **Gaussian Splats**: Require Blender addon for proper viewing
- **Memory Usage**: Models load once per server session for efficiency

## 🤝 Contributing

This integration maintains the original Make-It-Animatable functionality while providing ComfyUI compatibility. For core algorithm improvements, contribute to the main Make-It-Animatable repository.