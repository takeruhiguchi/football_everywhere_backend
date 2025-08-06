#!/usr/bin/env python3
"""
Test the existing completed job with workaround
"""

import requests
import os

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
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None

def test_existing_job():
    """Test existing job with workaround"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "44bc41f2-fca4-4035-923a-2c39cc6b198f"
    
    output_dir = "/home/takeru.higuchi/TextureGeneration/workflow_api/output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Testing existing job: {job_id}")
    
    # Get status
    response = requests.get(f"{base_url}/job/{job_id}/status")
    
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status['status']}")
        
        # WORKAROUND: Extract files from results
        files = status.get('files', [])
        if not files and status.get('results', {}).get('files'):
            print("🔧 Using workaround to extract files from results...")
            for file_info in status.get('results', {}).get('files', []):
                if file_info.get('type') not in ['image']:  # Skip temp images
                    filename = file_info.get('filename', file_info.get('path', '').split('/')[-1] if file_info.get('path') else '')
                    if filename:
                        files.append({
                            'type': file_info['type'],
                            'filename': filename,
                            'path': file_info.get('path', ''),
                            'format': file_info.get('format', '')
                        })
        
        print(f"📁 Files available: {len(files)}")
        
        # Show and download files
        downloaded_files = []
        for file_info in files:
            filename = file_info.get('filename', 'N/A')
            file_type = file_info.get('type', 'unknown')
            print(f"  📄 {file_type}: {filename}")
            
            if filename != 'N/A':
                downloaded_path = download_file(base_url, job_id, filename, output_dir)
                if downloaded_path:
                    downloaded_files.append(downloaded_path)
        
        print(f"\n✅ Successfully downloaded {len(downloaded_files)} files:")
        for file_path in downloaded_files:
            size = os.path.getsize(file_path)
            print(f"  - {file_path} ({size} bytes)")
    
    else:
        print(f"❌ Error: {response.status_code}")

if __name__ == "__main__":
    test_existing_job()