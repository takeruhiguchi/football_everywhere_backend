#!/usr/bin/env python3
"""
Test that API now returns only rigged character FBX files
"""

import requests
import os

def test_fbx_only():
    """Test that only FBX files are returned"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "44bc41f2-fca4-4035-923a-2c39cc6b198f"
    
    print(f"🔍 Testing FBX-only API with job {job_id}")
    
    # Get status
    response = requests.get(f"{base_url}/job/{job_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status['status']}")
        
        files = status.get('files', [])
        print(f"📁 Files returned: {len(files)}")
        
        # Show all files
        for i, file_info in enumerate(files):
            file_type = file_info.get('type', 'unknown')
            filename = file_info.get('filename', 'N/A')
            file_format = file_info.get('format', 'unknown')
            
            print(f"  File {i+1}: {file_type} - {filename} ({file_format})")
        
        # Verify only FBX files
        fbx_files = [f for f in files if f.get('format') == 'fbx' and f.get('type') == 'rigged_character']
        other_files = [f for f in files if not (f.get('format') == 'fbx' and f.get('type') == 'rigged_character')]
        
        print(f"\n✅ Rigged character FBX files: {len(fbx_files)}")
        print(f"❌ Other files: {len(other_files)}")
        
        if len(other_files) == 0:
            print("🎉 SUCCESS: Only rigged character FBX files are returned!")
        else:
            print("⚠️ WARNING: Non-FBX files are still being returned")
        
        # Test download of FBX file
        if fbx_files:
            fbx_file = fbx_files[0]
            filename = fbx_file.get('filename')
            
            print(f"\n📥 Testing download of: {filename}")
            
            output_dir = "/home/takeru.higuchi/TextureGeneration/workflow_api/output"
            os.makedirs(output_dir, exist_ok=True)
            
            download_url = f"{base_url}/job/{job_id}/download/{filename}"
            download_response = requests.get(download_url, stream=True)
            
            if download_response.status_code == 200:
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'wb') as f:
                    for chunk in download_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                size = os.path.getsize(file_path)
                print(f"✅ Downloaded: {file_path} ({size:,} bytes)")
            else:
                print(f"❌ Download failed: {download_response.status_code}")
    
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    test_fbx_only()