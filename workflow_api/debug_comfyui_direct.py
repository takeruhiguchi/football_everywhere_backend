#!/usr/bin/env python3
"""
Debug ComfyUI directly to see what results are available
"""

import urllib.request
import json
import requests

def debug_comfyui_history():
    """Check ComfyUI history directly"""
    
    server = "127.0.0.1:8188"
    
    print("🔍 Checking ComfyUI history...")
    
    try:
        # Get full history
        response = urllib.request.urlopen(f"http://{server}/history")
        history = json.loads(response.read())
        
        print(f"📊 Found {len(history)} prompts in history")
        
        # Show recent prompts
        recent_prompts = list(reversed(list(history.keys())))[:5]
        
        for i, prompt_id in enumerate(recent_prompts):
            print(f"\n📋 Prompt {i+1}: {prompt_id}")
            prompt_data = history[prompt_id]
            
            if "outputs" in prompt_data:
                outputs = prompt_data["outputs"]
                print(f"  Node outputs: {list(outputs.keys())}")
                
                for node_id, node_output in outputs.items():
                    print(f"    Node {node_id}: {list(node_output.keys())}")
                    
                    # Check for different output types
                    if "images" in node_output:
                        print(f"      Images: {len(node_output['images'])}")
                        for img in node_output["images"]:
                            print(f"        - {img.get('filename', 'N/A')}")
                    
                    if "glb_path" in node_output:
                        print(f"      GLB paths: {node_output['glb_path']}")
                    
                    if "result" in node_output:
                        print(f"      Results: {node_output['result']}")
                    
                    if "fbx_path" in node_output:
                        print(f"      FBX paths: {node_output['fbx_path']}")
                    
                    # Show all keys for debugging
                    print(f"      All keys: {list(node_output.keys())}")
            else:
                print("  No outputs found")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def debug_comfyui_outputs_dir():
    """Check what's in the ComfyUI output directory"""
    import os
    
    output_dir = "/home/takeru.higuchi/TextureGeneration/ComfyUI/output"
    
    print(f"📂 Checking output directory: {output_dir}")
    
    try:
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                rel_path = os.path.relpath(root, output_dir)
                if rel_path == ".":
                    print(f"  📁 / ({len(files)} files)")
                else:
                    print(f"  📁 {rel_path}/ ({len(files)} files)")
                
                # Show recent files (last 10)
                files.sort(key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True)
                for f in files[:10]:
                    file_path = os.path.join(root, f)
                    size = os.path.getsize(file_path)
                    print(f"    - {f} ({size} bytes)")
        else:
            print("  ❌ Output directory does not exist")
            
    except Exception as e:
        print(f"❌ Error checking output directory: {e}")

if __name__ == "__main__":
    debug_comfyui_history()
    print("\n" + "="*50)
    debug_comfyui_outputs_dir()