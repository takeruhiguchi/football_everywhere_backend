#!/usr/bin/env python3

from comfyui_client import ComfyUIClient
import json

client = ComfyUIClient('127.0.0.1:8188')

# Get recent history to find the prompt ID
try:
    import urllib.request
    response = urllib.request.urlopen('http://127.0.0.1:8188/history')
    all_history = json.loads(response.read())
    
    print('Recent ComfyUI history entries:')
    for prompt_id, history in list(all_history.items())[-3:]:  # Last 3 entries
        print(f'\nPrompt ID: {prompt_id}')
        if 'outputs' in history:
            print('  Outputs:')
            for node_id, outputs in history['outputs'].items():
                print(f'    Node {node_id}: {list(outputs.keys())}')
                if 'images' in outputs:
                    for img in outputs['images'][:2]:  # Show first 2 images
                        print(f'      Image: {img}')
                # Show any other output keys
                for key, value in outputs.items():
                    if key != 'images':
                        print(f'      {key}: {value}')
        else:
            print('  No outputs found')
            
except Exception as e:
    print(f'Error: {e}')