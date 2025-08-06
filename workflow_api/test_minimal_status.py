#!/usr/bin/env python3
"""
Test the minimal status response
"""

import requests
import json

def test_minimal_status():
    """Test that status response is now minimal and clean"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "6fec448d-c07f-4590-806f-128338ff14f2"  # Use the GCP job ID
    
    print(f"🔍 Testing minimal status response for job {job_id}")
    
    # Get status
    response = requests.get(f"{base_url}/job/{job_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        
        print("📊 Minimal Status Response:")
        print(json.dumps(status, indent=2))
        
        print(f"\n📈 Response size: {len(response.text)} characters")
        
        # Check what fields are included
        fields = list(status.keys())
        print(f"📋 Fields included: {', '.join(fields)}")
        
        # Verify no heavy fields
        heavy_fields = ['results']
        included_heavy = [f for f in heavy_fields if f in status]
        
        if included_heavy:
            print(f"⚠️ Warning: Heavy fields still included: {included_heavy}")
        else:
            print("✅ No heavy fields included - response is clean!")
        
        # Show file info if available
        files = status.get('files', [])
        if files:
            print(f"\n📁 Files ({len(files)}):")
            for file_info in files:
                print(f"  - {file_info.get('type', 'unknown')}: {file_info.get('filename', 'N/A')}")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_minimal_status()