#!/usr/bin/env python3
"""
Test the results extraction from ComfyUI history
"""

from comfyui_client import TextureGenerationAPI
import json

def test_results_extraction():
    """Test the get_results method with actual prompt IDs"""
    
    api = TextureGenerationAPI()
    
    # Test with the most recent prompt ID from the debug output
    prompt_id = "13558701-d506-444f-b6d7-875090b89442"
    
    print(f"🔍 Testing results extraction for prompt {prompt_id}")
    
    results = api.get_results(prompt_id)
    
    print("📊 Raw results:")
    print(json.dumps(results, indent=2))
    
    print(f"\n📁 Files found: {len(results.get('files', []))}")
    
    for i, file_info in enumerate(results.get('files', [])):
        print(f"  File {i+1}: {file_info}")
    
    # Also test the client method directly
    print(f"\n🔍 Testing direct history fetch for prompt {prompt_id}")
    history = api.client.get_history(prompt_id)
    
    if prompt_id in history:
        prompt_history = history[prompt_id]
        if "outputs" in prompt_history:
            outputs = prompt_history["outputs"]
            print(f"📊 Found {len(outputs)} node outputs:")
            
            for node_id, node_output in outputs.items():
                print(f"  Node {node_id}:")
                if "result" in node_output:
                    print(f"    Result: {node_output['result']}")
                    for i, item in enumerate(node_output['result']):
                        print(f"      Item {i}: {item} (type: {type(item).__name__})")

if __name__ == "__main__":
    test_results_extraction()