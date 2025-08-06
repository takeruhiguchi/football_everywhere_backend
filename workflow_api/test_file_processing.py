#!/usr/bin/env python3
"""
Test the file processing logic with real ComfyUI data
"""

import json
from comfyui_client import TextureGenerationAPI

# Initialize API
api = TextureGenerationAPI(
    "/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow_api.json",
    "127.0.0.1:8188"
)

# Get the most recent prompt ID from history
import urllib.request

response = urllib.request.urlopen('http://127.0.0.1:8188/history')
all_history = json.loads(response.read())
recent_prompt_id = list(all_history.keys())[-1]

print(f"Testing with prompt ID: {recent_prompt_id}")

# Get results using our updated logic
results = api.get_results(recent_prompt_id)

print(f"Status: {results['status']}")
print(f"Files found: {len(results.get('files', []))}")

# Focus on the non-image files (the important ones for download)
important_files = [f for f in results.get('files', []) if f['type'] != 'image']

print(f"\nImportant files for download:")
for file_info in important_files:
    print(f"  Type: {file_info['type']}")
    print(f"  Filename: {file_info.get('filename', 'N/A')}")
    print(f"  Path: {file_info.get('path', 'N/A')}")
    print(f"  Format: {file_info.get('format', 'N/A')}")
    
    # Test the file existence
    import os
    file_path = file_info.get('path', '')
    if file_path and not os.path.isabs(file_path):
        # Handle relative paths
        base_path = "/home/takeru.higuchi/TextureGeneration/ComfyUI/output"
        file_path = os.path.join(base_path, file_path)
    
    if file_path and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"  ✅ File exists: {file_path} ({file_size:,} bytes)")
    else:
        print(f"  ❌ File not found: {file_path}")
    print()

# Test the format that would be returned by the API
job_files = []
for file_info in results.get("files", []):
    filename = file_info.get("filename", file_info.get("path", "").split("/")[-1] if file_info.get("path") else "")
    if filename:
        job_files.append({
            "type": file_info["type"],
            "filename": filename,
            "path": file_info.get("path", ""),
            "format": file_info.get("format", "")
        })

print(f"Files that would be returned by API:")
for f in [jf for jf in job_files if jf['type'] != 'image']:  # Focus on important files
    print(f"  - {f['type']}: {f['filename']}")