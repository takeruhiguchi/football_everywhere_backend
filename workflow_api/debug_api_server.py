#!/usr/bin/env python3
"""
Debug the API server to understand why results are not being returned
"""

import requests
import json
import time

def debug_api_server():
    """Debug API server by creating a new job and monitoring it closely"""
    
    base_url = "http://127.0.0.1:8000"
    
    # Check if server is running
    try:
        health = requests.get(f"{base_url}/health")
        print(f"🔍 Server health: {health.status_code}")
        if health.status_code == 200:
            print(f"  Response: {health.json()}")
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        return
    
    # Test with a very simple request 
    test_image = "/home/takeru.higuchi/TextureGeneration/workflow_api/input/avatar_base.jpg"
    
    data = {
        "prompt": "test character",
        "steps": 5,  # Reduced for faster testing
        "enable_rigging": False  # Disable rigging for faster testing
    }
    
    print("🔍 Starting new generation job...")
    
    try:
        # Start generation
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/generate_character", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✅ Job started: {job_id}")
            
            # Monitor closely
            for i in range(60):  # Check for up to 10 minutes
                time.sleep(10)
                
                # Get status
                status_response = requests.get(f"{base_url}/job/{job_id}/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"📊 [{i*10}s] Status: {status['status']}, Progress: {status.get('progress', 0)}%")
                    
                    if status['status'] == 'completed':
                        print(f"🎉 Job completed!")
                        print(f"📁 Files in status: {len(status.get('files', []))}")
                        
                        # Get debug info
                        debug_response = requests.get(f"{base_url}/debug/job/{job_id}")
                        if debug_response.status_code == 200:
                            debug_info = debug_response.json()
                            print("🐛 Debug info:")
                            print(json.dumps(debug_info, indent=2))
                        
                        # Show files
                        for file_info in status.get('files', []):
                            print(f"  📄 {file_info['type']}: {file_info.get('filename', 'NO NAME')}")
                        
                        break
                    elif status['status'] in ['failed', 'timeout']:
                        print(f"❌ Job failed: {status.get('error', 'Unknown')}")
                        break
                else:
                    print(f"❌ Error getting status: {status_response.status_code}")
                    break
        else:
            print(f"❌ Failed to start job: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_api_server()