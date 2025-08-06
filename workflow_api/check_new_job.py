#!/usr/bin/env python3

import requests, json

job_id = "eb761fbb-aead-4d03-8c66-4002ad9408d3"
response = requests.get(f"http://127.0.0.1:8000/job/{job_id}/status")
data = response.json()

files = data.get('files', [])
print(f'Total files: {len(files)}')
print('File types:', [f.get('type') for f in files])

non_image_files = []
for f in files:
    if f.get('type') != 'image':
        non_image_files.append(f)

print(f'Non-image files: {len(non_image_files)}')
for f in non_image_files:
    print(f'  - Type: {f.get("type", "unknown")}')
    print(f'    Filename: {f.get("filename", "no-filename")}')
    print(f'    Path: {f.get("path", "no-path")}')
    print(f'    Format: {f.get("format", "no-format")}')
    print()