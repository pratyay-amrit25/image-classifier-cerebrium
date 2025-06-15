import requests
import argparse
import time
from argparse import Namespace
from typing import Dict, Any, List, Tuple

def test_endpoint(image_path: str, api_key: str, endpoint_url: str) -> None:
    # Adjust endpoint URL for local testing (append /predict if needed)
    if "localhost" in endpoint_url and not endpoint_url.endswith("/predict"):
        endpoint_url = f"{endpoint_url}/predict"
    
    # Use X-API-Key header (Cerebrium style), or skip if not needed locally
    headers: Dict[str, str] = {"X-API-Key": api_key}
    with open(image_path, "rb") as image_file:
        files: Dict[str, Tuple[str, Any, str]] = {"image": (image_path, image_file, "image/jpeg")}
        start_time = time.time()
        response = requests.post(endpoint_url, headers=headers, files=files)
        latency = time.time() - start_time
        print(f"Latency: {latency:.2f} seconds")
        if response.status_code == 200:
            result = response.json().get("my_result", {})
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Class ID: {result.get('class_id')}, Probabilities: {result.get('probabilities', [])[:5]}...")
        else:
            print(f"Failed with status {response.status_code}: {response.text}")

def run_custom_tests(api_key: str, endpoint_url: str) -> None:
    tests: List[Tuple[str, int]] = [
        ("images/n01440764_tench.jpg", 0),
        ("images/n01667114_mud_turtle.jpg", 35)
    ]
    for image_path, expected_class in tests:
        print(f"\nTesting {image_path} (Expected Class: {expected_class})")
        test_endpoint(image_path, api_key, endpoint_url)
    
    # Health check (only for Cerebrium endpoint, skip for local)
    if "localhost" not in endpoint_url:
        health_url = endpoint_url.replace("/run", "/health")
        start_time = time.time()
        response = requests.get(health_url, headers={"X-API-Key": api_key})
        latency = time.time() - start_time
        print(f"\nHealth Check Latency: {latency:.2f} seconds")
        print(f"Health Check Status: {response.status_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cerebrium deployed model")
    parser.add_argument("--image", help="Path to test image")
    # Note: The API key and endpoint URL for a deployed Cerebrium model can be found on your Cerebrium dashboard.
    parser.add_argument("--api-key", required=True, help="API key (use dummy-key for local testing)")
    parser.add_argument("--endpoint", required=True, help="Endpoint URL (e.g., http://localhost:8000 or Cerebrium URL)")
    parser.add_argument("--custom-tests", action="store_true", help="Run custom tests")
    args: Namespace = parser.parse_args()

    if args.custom_tests:
        run_custom_tests(args.api_key, args.endpoint)
    elif args.image:
        test_endpoint(args.image, args.api_key, args.endpoint)
    else:
        print("Please provide --image or --custom-tests")