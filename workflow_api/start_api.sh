#!/bin/bash

# ComfyUI Texture Generation API Startup Script

set -e

echo "🚀 Starting ComfyUI Texture Generation API"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: api_server.py not found in current directory"
    echo "   Please run this script from the workflow_api directory"
    exit 1
fi

# Check if Python requirements are installed
echo "📦 Checking Python dependencies..."
if ! python -c "import fastapi, uvicorn, requests, websocket, PIL" 2>/dev/null; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
else
    echo "✅ All dependencies are installed"
fi

# Check if ComfyUI is running
echo "🔍 Checking ComfyUI server connection..."
if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "✅ ComfyUI server is running"
else
    echo "⚠️  ComfyUI server is not running or not accessible"
    echo "   Please start ComfyUI first:"
    echo "   cd /home/takeru.higuchi/TextureGeneration/ComfyUI"
    echo "   python main.py"
    echo ""
    echo "   Then run this script again."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if workflow file exists
WORKFLOW_PATH="/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow.json"
if [ ! -f "$WORKFLOW_PATH" ]; then
    echo "❌ Error: Workflow file not found at $WORKFLOW_PATH"
    echo "   Please ensure the workflow file exists"
    exit 1
else
    echo "✅ Workflow file found"
fi

# Create upload directory if it doesn't exist
mkdir -p /tmp/comfyui_uploads
echo "✅ Upload directory ready"

echo ""
echo "🌟 Starting API server..."
echo "   API will be available at: http://localhost:8000"
echo "   Swagger UI will be available at: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the API server
python api_server.py