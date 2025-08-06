#!/usr/bin/env python3
"""
Test the fixed API with existing completed job
"""

import requests
import json

def test_fixed_api():
    """Test if the API now returns files correctly"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "44bc41f2-fca4-4035-923a-2c39cc6b198f"
    
    print(f"🔍 Testing fixed API with job {job_id}")
    
    # Get job status
    response = requests.get(f"{base_url}/job/{job_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status['status']}")
        print(f"📁 Files count: {len(status.get('files', []))}")
        
        # Show non-image files
        files = status.get('files', [])
        non_image_files = [f for f in files if f.get('type') != 'image']
        
        print(f"📄 Non-image files: {len(non_image_files)}")
        for file_info in non_image_files:
            print(f"  - {file_info.get('type', 'unknown')}: {file_info.get('filename', 'NO NAME')}")
        
        # Test downloading the first non-image file
        if non_image_files:
            test_file = non_image_files[0]
            filename = test_file.get('filename')
            
            if filename:
                print(f"\n📥 Testing download: {filename}")
                download_url = f"{base_url}/job/{job_id}/download/{filename}"
                
                try:
                    download_response = requests.head(download_url)  # Just check if file exists
                    print(f"  Status: {download_response.status_code}")
                    if download_response.status_code == 200:
                        content_length = download_response.headers.get('Content-Length', 'unknown')
                        print(f"  File size: {content_length} bytes")
                        print(f"✅ Download endpoint working!")
                    else:
                        print(f"❌ Download failed: {download_response.status_code}")
                except Exception as e:
                    print(f"❌ Download error: {e}")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_fixed_api()