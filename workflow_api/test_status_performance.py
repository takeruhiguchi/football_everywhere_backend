#!/usr/bin/env python3
"""
Test status API performance
"""

import requests
import time

def test_status_performance():
    """Test how fast the status API responds"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "6fec448d-c07f-4590-806f-128338ff14f2"
    
    print(f"🔍 Testing status API performance for job {job_id}")
    
    # Test multiple requests to see caching effect
    times = []
    
    for i in range(5):
        print(f"\n📊 Request {i+1}/5:")
        
        start_time = time.time()
        response = requests.get(f"{base_url}/job/{job_id}/status")
        end_time = time.time()
        
        response_time = end_time - start_time
        times.append(response_time)
        
        if response.status_code == 200:
            status = response.json()
            files_count = len(status.get('files', []))
            response_size = len(response.text)
            
            print(f"  ⏱️  Response time: {response_time:.3f}s")
            print(f"  📁 Files: {files_count}")
            print(f"  📏 Response size: {response_size} chars")
            print(f"  ✅ Status: {status['status']}")
        else:
            print(f"  ❌ Error: {response.status_code}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Performance summary
    print(f"\n📈 Performance Summary:")
    print(f"  Average response time: {sum(times)/len(times):.3f}s")
    print(f"  Min response time: {min(times):.3f}s")
    print(f"  Max response time: {max(times):.3f}s")
    
    if max(times) > 2.0:
        print("  ⚠️  Warning: Some responses are slow (>2s)")
    elif max(times) > 1.0:
        print("  ⚡ Good: Responses under 2s")
    else:
        print("  🚀 Excellent: All responses under 1s")

def test_simple_health():
    """Test simple health endpoint for comparison"""
    
    base_url = "http://127.0.0.1:8000"
    
    print(f"\n🏥 Testing health endpoint for comparison:")
    
    start_time = time.time()
    response = requests.get(f"{base_url}/health")
    end_time = time.time()
    
    response_time = end_time - start_time
    
    if response.status_code == 200:
        print(f"  ⏱️  Health response time: {response_time:.3f}s")
        print("  ✅ Health check passed")
    else:
        print(f"  ❌ Health check failed: {response.status_code}")

if __name__ == "__main__":
    test_status_performance()
    test_simple_health()