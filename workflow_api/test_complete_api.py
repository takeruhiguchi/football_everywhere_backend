#!/usr/bin/env python3
"""
Test the complete 3D character generation API
"""

import requests
import json
import time
import os
from pathlib import Path

def download_file(base_url, job_id, filename, output_dir):
    """Download a file from the API and save it to the output directory"""
    try:
        download_url = f"{base_url}/job/{job_id}/download/{filename}"
        response = requests.get(download_url, stream=True)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {filename}")
            return file_path
        else:
            print(f"❌ Failed to download {filename}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None

def test_complete_api():
    print("🚀 Testing complete 3D character generation API...")
    
    # API endpoint
    base_url = "http://127.0.0.1:8000"
    
    # Test image path
    test_image = "/home/takeru.higuchi/TextureGeneration/workflow_api/input/avatar_base.jpg"
    
    # Create output directory
    output_dir = "/home/takeru.higuchi/TextureGeneration/workflow_api/output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Test parameters
        data = {
            "prompt": "3D anime character with detailed armor",
            "negative_prompt": "low quality, blurry, deformed",
            "image_size": 768,
            "mesh_input_size": 1024,
            "outline_size": 128,
            "steps": 10,
            "guidance_scale": 5.5,
            "cfg_scale": 2.0,
            "seed": 42,
            "model_name": "dreamshaper.safetensors",
            "pose_type": "大-pose",
            "enable_rigging": True
        }
        
        print("📤 Uploading image and starting generation...")
        
        # Upload file and start generation
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/generate_character", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✅ Generation started successfully! Job ID: {job_id}")
            
            # Monitor job status
            print("⏳ Monitoring job progress...")
            start_time = time.time()
            
            while True:
                status_response = requests.get(f"{base_url}/job/{job_id}/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"📊 Status: {status['status']}")
                    
                    if status['status'] == 'completed':
                        print("🎉 Generation completed successfully!")
                        
                        # WORKAROUND: If files is empty, extract only rigged character FBX from results
                        files = status.get('files', [])
                        if not files and status.get('results', {}).get('files'):
                            print("🔧 Using workaround to extract rigged character FBX from results...")
                            for file_info in status.get('results', {}).get('files', []):
                                # Only include rigged character FBX files
                                if file_info.get('type') == 'rigged_character' and file_info.get('format') == 'fbx':
                                    filename = file_info.get('filename', file_info.get('path', '').split('/')[-1] if file_info.get('path') else '')
                                    if filename:
                                        files.append({
                                            'type': file_info['type'],
                                            'filename': filename,
                                            'path': file_info.get('path', ''),
                                            'format': file_info.get('format', '')
                                        })
                        
                        print(f"📁 Files generated: {len(files)}")
                        
                        # Download all generated files
                        print("📥 Downloading generated files...")
                        downloaded_files = []
                        
                        for file_info in files:
                            filename = file_info.get('filename', file_info.get('path', 'N/A'))
                            if filename != 'N/A':
                                print(f"  - {file_info['type']}: {filename}")
                                downloaded_path = download_file(base_url, job_id, filename, output_dir)
                                if downloaded_path:
                                    downloaded_files.append(downloaded_path)
                        
                        print(f"✅ Downloaded {len(downloaded_files)} files to {output_dir}")
                        print("📂 Downloaded files:")
                        for file_path in downloaded_files:
                            print(f"  - {file_path}")
                        
                        break
                    elif status['status'] == 'failed':
                        print(f"❌ Generation failed: {status.get('error', 'Unknown error')}")
                        break
                    elif status['status'] == 'timeout':
                        print("⏰ Generation timed out")
                        break
                    elif time.time() - start_time > 600:  # 10 minute timeout
                        print("⏰ Test timeout reached")
                        break
                    
                    time.sleep(10)  # Check every 10 seconds
                else:
                    print(f"❌ Error checking status: {status_response.text}")
                    break
                    
        else:
            print(f"❌ Generation request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during API test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_api()