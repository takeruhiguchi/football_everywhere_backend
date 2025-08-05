# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a texture generation project built on ComfyUI, an advanced node-based interface for Stable Diffusion and AI model workflows. The project includes multiple custom nodes for 3D processing, image/video generation, and advanced AI capabilities.

## Core Architecture

**ComfyUI Core**: The main framework at `/ComfyUI/` provides:
- Node-based visual workflow system for AI model pipelines
- Support for image, video, audio, and 3D model generation
- Extensible custom node system
- API server for external integrations
- Database-backed user and workflow management

**Custom Nodes**: Extended functionality through specialized nodes:
- **Make-It-Animatable**: 3D character animation and rigging from static models
- **ComfyUI-Hunyuan3DWrapper**: Text/image to 3D model generation
- **ComfyUI-MVAdapter**: Multi-view image generation for 3D workflows
- **ComfyUI-3D-Convert**: 3D format conversion utilities
- **ComfyUI-Manager**: Package management for custom nodes

## Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install with uv (recommended)
uv install
```

### Running ComfyUI
```bash
# Start the main server
python main.py

# With specific GPU/device options
python main.py --cuda-device 0
python main.py --cpu  # CPU-only mode
python main.py --preview-method taesd  # High-quality previews
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not inference"  # Skip inference tests
pytest -m "not execution"  # Skip execution tests
pytest tests-unit/         # Unit tests only
pytest tests/              # Integration tests only
```

### Development Tools
```bash
# Code formatting and linting
ruff check .
ruff format .

# Database migrations
alembic upgrade head
```

## Key Components

### Core Modules
- **`main.py`**: Main server entry point and CLI argument handling
- **`nodes.py`**: Built-in node definitions for core functionality  
- **`execution.py`**: Workflow execution engine and queue management
- **`server.py`**: Web server and API endpoints
- **`comfy/`**: Core model management, sampling, and inference
- **`comfy_execution/`**: Graph processing and caching system

### Custom Node Integration
- **Node Discovery**: Custom nodes in `custom_nodes/` are auto-loaded on startup
- **Installation**: Use ComfyUI-Manager for package management
- **API Integration**: Isolated environments for complex dependencies (e.g., Make-It-Animatable)

### Model Management
- **Model Paths**: Configured via `extra_model_paths.yaml` for shared model libraries
- **Supported Formats**: `.ckpt`, `.safetensors`, `.pt`, `.pth` checkpoints
- **Model Types**: Checkpoints, VAE, LoRA, ControlNet, embeddings organized in `models/` subdirectories

## Configuration

### Model Storage Structure
```
models/
├── checkpoints/        # Main diffusion models
├── vae/               # VAE models  
├── loras/             # LoRA adapters
├── controlnet/        # ControlNet models
├── upscale_models/    # Upscaling models
├── embeddings/        # Textual inversions
└── diffusion_models/  # Additional diffusion models
```

### Server Configuration
- **Default Port**: 8188 (web interface)
- **API Access**: RESTful API and WebSocket support
- **TLS/SSL**: Configurable with `--tls-keyfile` and `--tls-certfile`
- **Frontend**: Separate frontend repository with version selection via `--front-end-version`

## Development Notes

### Testing Strategy
- **Unit Tests**: Fast tests in `tests-unit/` for isolated functionality
- **Integration Tests**: Full workflow tests in `tests/` with model loading
- **Markers**: Use pytest markers to control test execution scope
- **Performance**: Integration tests may require GPU and significant time

### Memory Management
- **Smart Offloading**: Automatic GPU memory management for low-VRAM systems
- **Model Caching**: Intelligent caching to avoid redundant model loads
- **Batch Processing**: Queue system for efficient batch processing

### Custom Node Development
- **Entry Point**: `__init__.py` with `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`
- **Dependencies**: Use `requirements.txt` or `pyproject.toml` for node-specific dependencies
- **API Isolation**: Complex nodes should use separate API servers to avoid dependency conflicts

### 3D Workflow Integration
- **Make-It-Animatable**: Character rigging and animation via isolated FastAPI server
- **3D Formats**: GLB, FBX, OBJ support through specialized conversion nodes
- **Blender Integration**: Some nodes use Blender's `bpy` module for advanced 3D processing

## Current Status

### Functional Components ✅
- **Core ComfyUI**: Fully operational with extensive model support
- **3D Generation**: Hunyuan3D and multi-view generation working
- **Custom Nodes**: Most extensions functional and integrated
- **API Services**: External API integrations operational

### Known Issues ❌
- **Make-It-Animatable**: Blender integration has segmentation fault in rigging pipeline
- **Memory Usage**: High memory requirements for 3D and video models
- **Frontend Updates**: Frontend repository separate from core, may have version mismatches