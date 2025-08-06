#!/usr/bin/env python3

import urllib.request, json
from comfyui_client import TextureGenerationAPI

# Get the most recent ComfyUI history
response = urllib.request.urlopen('http://127.0.0.1:8188/history')
history = json.loads(response.read())

print('Most recent 3 prompt IDs from ComfyUI:')
for i, prompt_id in enumerate(list(history.keys())[-3:]):
    print(f'{i+1}. {prompt_id}')

# Test our get_results function with the latest prompt_id
api = TextureGenerationAPI(
    '/home/takeru.higuchi/TextureGeneration/ComfyUI/user/default/workflows/main_workflow_api.json',
    '127.0.0.1:8188'
)

latest_prompt_id = list(history.keys())[-1]
print(f'\nTesting get_results with latest prompt_id: {latest_prompt_id}')

results = api.get_results(latest_prompt_id)
print(f'Status: {results.get("status", "no-status")}')
print(f'Files found: {len(results.get("files", []))}')

non_image_files = [f for f in results.get('files', []) if f.get('type') != 'image']
print(f'Non-image files: {len(non_image_files)}')
for f in non_image_files:
    print(f'  - {f.get("type", "unknown")}: {f.get("filename", "no-filename")}')
    print(f'    Path: {f.get("path", "")}')
    print()