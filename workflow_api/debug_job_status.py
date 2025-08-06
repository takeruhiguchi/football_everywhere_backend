#!/usr/bin/env python3
"""
Debug job status to understand why files are not being returned
"""

import requests
import json

def debug_job_status():
    """Debug a completed job to see why files are not returned"""
    
    base_url = "http://127.0.0.1:8000"
    
    # Get the most recent job ID from the test output or replace with a known job ID
    job_id = "5fe3b9ed-2ccc-4e31-b13c-b2c5addf004e"  # Replace with actual job ID
    
    print(f"🔍 Debugging job {job_id}...")
    
    # Get job status
    status_response = requests.get(f"{base_url}/job/{job_id}/status")
    if status_response.status_code == 200:
        status = status_response.json()
        print("📊 Job Status Response:")
        print(json.dumps(status, indent=2))
        
        print(f"\n📁 Files count: {len(status.get('files', []))}")
        for i, file_info in enumerate(status.get('files', [])):
            print(f"  File {i+1}: {file_info}")
    else:
        print(f"❌ Error getting status: {status_response.status_code}")
        print(status_response.text)
    
    # Get debug information
    debug_response = requests.get(f"{base_url}/debug/job/{job_id}")
    if debug_response.status_code == 200:
        debug_info = debug_response.json()
        print("\n🐛 Debug Information:")
        print(json.dumps(debug_info, indent=2))
    else:
        print(f"❌ Error getting debug info: {debug_response.status_code}")

if __name__ == "__main__":
    debug_job_status()