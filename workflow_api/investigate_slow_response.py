#!/usr/bin/env python3
"""
Investigate potential causes of slow status responses
"""

import requests
import time
import concurrent.futures

def test_concurrent_requests():
    """Test if concurrent requests cause slowdowns"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "6fec448d-c07f-4590-806f-128338ff14f2"
    
    print("🔍 Testing concurrent requests:")
    
    def single_request(request_id):
        start_time = time.time()
        try:
            response = requests.get(f"{base_url}/job/{job_id}/status", timeout=10)
            end_time = time.time()
            return {
                'id': request_id,
                'time': end_time - start_time,
                'status': response.status_code,
                'success': True
            }
        except Exception as e:
            end_time = time.time()
            return {
                'id': request_id,
                'time': end_time - start_time,
                'error': str(e),
                'success': False
            }
    
    # Test 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(single_request, i) for i in range(10)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Analyze results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    if successful:
        times = [r['time'] for r in successful]
        print(f"  ✅ Successful requests: {len(successful)}")
        print(f"  ⏱️  Average time: {sum(times)/len(times):.3f}s")
        print(f"  ⏱️  Max time: {max(times):.3f}s")
        print(f"  ⏱️  Min time: {min(times):.3f}s")
    
    if failed:
        print(f"  ❌ Failed requests: {len(failed)}")
        for r in failed:
            print(f"    Request {r['id']}: {r.get('error', 'unknown error')}")

def test_different_job_states():
    """Test if job state affects response time"""
    
    base_url = "http://127.0.0.1:8000"
    
    print("\n🔍 Testing response times for different scenarios:")
    
    # Test health endpoint (baseline)
    start_time = time.time()
    response = requests.get(f"{base_url}/health")
    health_time = time.time() - start_time
    print(f"  Health endpoint: {health_time:.3f}s")
    
    # Test models endpoint
    start_time = time.time()
    response = requests.get(f"{base_url}/models")
    models_time = time.time() - start_time
    print(f"  Models endpoint: {models_time:.3f}s")
    
    # Test non-existent job (should be fast)
    start_time = time.time()
    response = requests.get(f"{base_url}/job/non-existent/status")
    nonexistent_time = time.time() - start_time
    print(f"  Non-existent job: {nonexistent_time:.3f}s (expected 404)")
    
def test_network_conditions():
    """Test different timeout scenarios"""
    
    base_url = "http://127.0.0.1:8000"
    job_id = "6fec448d-c07f-4590-806f-128338ff14f2"
    
    print("\n🔍 Testing with different timeout conditions:")
    
    timeouts = [1, 2, 5, 10]  # seconds
    
    for timeout in timeouts:
        try:
            start_time = time.time()
            response = requests.get(f"{base_url}/job/{job_id}/status", timeout=timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                print(f"  Timeout {timeout}s: {response_time:.3f}s ✅")
            else:
                print(f"  Timeout {timeout}s: {response_time:.3f}s ❌ ({response.status_code})")
                
        except requests.exceptions.Timeout:
            print(f"  Timeout {timeout}s: TIMED OUT ⏰")
        except Exception as e:
            print(f"  Timeout {timeout}s: ERROR {e}")

if __name__ == "__main__":
    test_concurrent_requests()
    test_different_job_states()
    test_network_conditions()