#!/usr/bin/env python3

from comfyui_client import TextureGenerationAPI

# Test the updated file extraction logic
api = TextureGenerationAPI(
    "/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow_api.json",
    "127.0.0.1:8188"
)

# Get the most recent prompt ID from history
import urllib.request
import json

response = urllib.request.urlopen('http://127.0.0.1:8188/history')
all_history = json.loads(response.read())

# Get the most recent prompt ID
recent_prompt_id = list(all_history.keys())[-1]
print(f"Testing file extraction for prompt ID: {recent_prompt_id}")

# Test the updated get_results method
results = api.get_results(recent_prompt_id)

print(f"\nExtracted results:")
print(f"Status: {results['status']}")
print(f"Files found: {len(results.get('files', []))}")

for file_info in results.get('files', []):
    print(f"  - Type: {file_info['type']}")
    print(f"    Filename: {file_info.get('filename', 'N/A')}")
    print(f"    Path: {file_info.get('path', 'N/A')}")
    print(f"    Format: {file_info.get('format', 'N/A')}")
    print()