#!/usr/bin/env python3

import requests
import json

# Get the job status
job_id = "c908daa9-b41a-4ffc-845b-77cb8f48a850"
response = requests.get(f"http://127.0.0.1:8000/job/{job_id}/status")

if response.status_code == 200:
    job_data = response.json()
    
    print("=== Job Data Analysis ===")
    print(f"Job ID: {job_data['job_id']}")
    print(f"Status: {job_data['status']}")
    print(f"Files count: {len(job_data.get('files', []))}")
    
    # Print files info
    files = job_data.get('files', [])
    important_files = [f for f in files if f.get('type') not in ['image']]
    
    print(f"\nImportant files (non-image): {len(important_files)}")
    for f in important_files:
        print(f"  - {f.get('type', 'unknown')}: {f.get('filename', 'no filename')}")
    
    # Check if results contain more detailed info
    results = job_data.get('results', {})
    print(f"\nResults status: {results.get('status', 'no status')}")
    print(f"Results files count: {len(results.get('files', []))}")
    
    # Look for the prompt_id in results
    print(f"Has outputs in results: {'outputs' in results}")
    
    # Try to find the job in the global jobs data structure
    print(f"\n=== Missing Data ===")
    print(f"Missing: prompt_id in job data")
    
else:
    print(f"Error getting job: {response.status_code} {response.text}")

# Also check the most recent ComfyUI history for comparison
print(f"\n=== ComfyUI Recent History ===")
import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8188/history')
    history = json.loads(response.read())
    
    recent_prompt = list(history.keys())[-1]
    print(f"Most recent prompt_id: {recent_prompt}")
    
    # Check if this prompt has the files we expect
    hist = history[recent_prompt]
    if 'outputs' in hist:
        for node_id, outputs in hist['outputs'].items():
            if 'result' in outputs:
                results = outputs['result']
                for r in results:
                    if r and isinstance(r, str):
                        if r.endswith('.fbx') or r.endswith('.glb'):
                            print(f"  Found {r.split('.')[-1].upper()}: {r.split('/')[-1]}")
except Exception as e:
    print(f"Error checking ComfyUI: {e}")