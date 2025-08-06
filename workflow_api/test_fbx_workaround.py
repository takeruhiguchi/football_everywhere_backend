#!/usr/bin/env python3
"""
Test FBX-only extraction with client-side workaround
"""

import requests
import os

def test_fbx_workaround():
    """Test FBX-only extraction with workaround"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "44bc41f2-fca4-4035-923a-2c39cc6b198f"
    
    print(f"🔍 Testing FBX-only extraction with job {job_id}")
    
    # Get status
    response = requests.get(f"{base_url}/job/{job_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status['status']}")
        
        # Apply FBX-only filter with workaround
        fbx_files = []
        
        # First try from job["files"] if available
        files = status.get('files', [])
        for file_info in files:
            if file_info.get('type') == 'rigged_character' and file_info.get('format') == 'fbx':
                fbx_files.append(file_info)
        
        # If no files found, extract from results (workaround)
        if not fbx_files and status.get('results', {}).get('files'):
            print("🔧 Using workaround to extract rigged character FBX from results...")
            for file_info in status.get('results', {}).get('files', []):
                # Only include rigged character FBX files
                if file_info.get('type') == 'rigged_character' and file_info.get('format') == 'fbx':
                    filename = file_info.get('filename', file_info.get('path', '').split('/')[-1] if file_info.get('path') else '')
                    if filename:
                        fbx_files.append({
                            'type': file_info['type'],
                            'filename': filename,
                            'path': file_info.get('path', ''),
                            'format': file_info.get('format', '')
                        })
        
        print(f"📁 Rigged character FBX files found: {len(fbx_files)}")
        
        # Show and download FBX files only
        output_dir = "/home/takeru.higuchi/TextureGeneration/workflow_api/output"
        os.makedirs(output_dir, exist_ok=True)
        
        downloaded_files = []
        for file_info in fbx_files:
            filename = file_info.get('filename', 'N/A')
            file_type = file_info.get('type', 'unknown')
            print(f"  📄 {file_type}: {filename}")
            
            if filename != 'N/A':
                # Download file
                download_url = f"{base_url}/job/{job_id}/download/{filename}"
                download_response = requests.get(download_url, stream=True)
                
                if download_response.status_code == 200:
                    file_path = os.path.join(output_dir, filename)
                    with open(file_path, 'wb') as f:
                        for chunk in download_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    size = os.path.getsize(file_path)
                    print(f"  ✅ Downloaded: {filename} ({size:,} bytes)")
                    downloaded_files.append(file_path)
                else:
                    print(f"  ❌ Failed to download {filename}: {download_response.status_code}")
        
        print(f"\n🎉 Successfully downloaded {len(downloaded_files)} FBX file(s):")
        for file_path in downloaded_files:
            size = os.path.getsize(file_path)
            print(f"  - {os.path.basename(file_path)} ({size:,} bytes)")
    
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    test_fbx_workaround()