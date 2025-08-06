#!/usr/bin/env python3

import requests, json

response = requests.get('http://127.0.0.1:8000/job/c908daa9-b41a-4ffc-845b-77cb8f48a850/status')
job = response.json()
results = job.get('results', {})
files = results.get('files', [])

print(f'Total files in results: {len(files)}')
print('File types:', [f.get('type', 'no-type') for f in files])

# Look for non-image files
non_image_files = [f for f in files if f.get('type') != 'image']
print(f'\nNon-image files: {len(non_image_files)}')
for f in non_image_files:
    print(f'  - Type: {f.get("type", "unknown")}')
    print(f'    Path: {f.get("path", "no-path")}')
    print(f'    Filename: {f.get("filename", "no-filename")}')
    print()

# Check if any files have the 'filename' field populated
files_with_filename = [f for f in files if f.get('filename')]
print(f'Files with filename field: {len(files_with_filename)}')

# Check if any files have the 'path' field populated
files_with_path = [f for f in files if f.get('path')]
print(f'Files with path field: {len(files_with_path)}')

# Show first few files to understand structure
print('\nFirst 3 files structure:')
for i, f in enumerate(files[:3]):
    print(f'File {i+1}: {json.dumps(f, indent=2)}')