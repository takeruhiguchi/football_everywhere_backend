"""
Example usage of the ComfyUI Texture Generation API
"""

import requests
import time
import json
from pathlib import Path


def generate_character_example():
    """Example of using the API to generate a 3D character."""
    
    # API endpoint
    api_base = "http://localhost:8000"
    
    # Example image path (replace with your actual image)
    image_path = "/home/takeru.higuchi/TextureGeneration/ComfyUI/input/avatar_base.jpg"
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found at {image_path}")
        print("Please update the image_path variable with a valid image file.")
        return
    
    print("🚀 Starting character generation...")
    
    # Prepare the request
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        data = {
            'prompt': 'anime character with detailed clothing and armor',
            'negative_prompt': 'low quality, blurry, distorted, ugly',
            'image_size': 768,
            'steps': 15,
            'guidance_scale': 6.0,
            'seed': 42,
            'enable_rigging': True
        }
        
        # Submit the generation request
        response = requests.post(f"{api_base}/generate_character", 
                               files=files, 
                               data=data)
    
    if response.status_code != 200:
        print(f"❌ Error submitting request: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    job_id = result['job_id']
    print(f"✅ Job submitted successfully!")
    print(f"📋 Job ID: {job_id}")
    print(f"⏱️  Estimated time: {result['estimated_time']}")
    
    # Poll for completion
    print("\n🔄 Monitoring job progress...")
    while True:
        status_response = requests.get(f"{api_base}/job/{job_id}/status")
        
        if status_response.status_code != 200:
            print(f"❌ Error checking status: {status_response.status_code}")
            break
        
        status = status_response.json()
        
        print(f"\r📊 Status: {status['status']} | Progress: {status.get('progress', 0)}% | Stage: {status.get('current_stage', 'unknown')}", end="")
        
        if status['status'] == 'completed':
            print("\n🎉 Generation completed!")
            print(f"⏱️  Processing time: {status.get('processing_time', 'unknown')}")
            
            if 'results' in status and status['results']:
                print("\n📁 Generated files:")
                for file_info in status['results'].get('files', []):
                    if file_info['type'] == '3d_model':
                        print(f"  🗿 3D Model: {file_info['path']}")
                    elif file_info['type'] == 'image':
                        print(f"  🖼️  Image: {file_info['filename']}")
            
            break
        
        elif status['status'] == 'failed':
            print(f"\n❌ Generation failed!")
            print(f"   Error: {status.get('error', 'Unknown error')}")
            if 'failed_stage' in status:
                print(f"   Failed at stage: {status['failed_stage']}")
            break
        
        elif status['status'] == 'processing':
            # Show detailed stage information
            if 'stages' in status and status['stages']:
                print(f"\n📈 Detailed progress:")
                for stage in status['stages']:
                    status_icon = "✅" if stage['status'] == 'completed' else "🔄" if stage['status'] == 'processing' else "⏳"
                    print(f"   {status_icon} {stage['name']}: {stage['status']}")
        
        time.sleep(5)  # Check every 5 seconds


def test_api_health():
    """Test if the API is running and healthy."""
    
    api_base = "http://localhost:8000"
    
    try:
        response = requests.get(f"{api_base}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is healthy")
            print(f"   Status: {health['status']}")
            print(f"   ComfyUI Server: {health['comfyui_server']}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
        print("   Make sure the API server is running on http://localhost:8000")
        return False


def list_available_models():
    """List available models."""
    
    api_base = "http://localhost:8000"
    
    try:
        response = requests.get(f"{api_base}/models")
        
        if response.status_code == 200:
            models = response.json()
            print("🎨 Available Models:")
            print("\n  Diffusion Models:")
            for model in models.get('diffusion_models', []):
                print(f"    • {model}")
            
            print("\n  3D Models:")
            for model in models.get('3d_models', []):
                print(f"    • {model}")
        else:
            print(f"❌ Error fetching models: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to API: {e}")


def main():
    """Main example function."""
    
    print("🎮 ComfyUI Texture Generation API - Example Usage")
    print("=" * 50)
    
    # Test API health
    print("\n1️⃣  Testing API connection...")
    if not test_api_health():
        print("\n💡 To start the API server, run:")
        print("   cd /home/takeru.higuchi/TextureGeneration/workflow_api")
        print("   python api_server.py")
        return
    
    # List available models
    print("\n2️⃣  Fetching available models...")
    list_available_models()
    
    # Generate character
    print("\n3️⃣  Starting character generation...")
    generate_character_example()


if __name__ == "__main__":
    main()