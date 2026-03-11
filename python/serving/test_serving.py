#!/usr/bin/env python3
"""Test script for MedGemma simple serving."""

import base64
import io
import json
import requests
from PIL import Image

SERVER_URL = "http://localhost:8080"


def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_simple_predict():
    """Test simple predict endpoint with text only."""
    print("Testing /predict endpoint (text only)...")
    payload = {
        "prompt": "What is medical imaging? Please explain briefly.",
        "max_tokens": 200,
        "temperature": 0.7
    }
    response = requests.post(f"{SERVER_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_chat_completion_text():
    """Test chat completions endpoint with text only."""
    print("Testing /v1/chat/completions (text only)...")
    payload = {
        "messages": [
            {"role": "user", "content": "What is a chest X-ray used for?"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']}")
    print()


def test_chat_completion_with_image():
    """Test chat completions endpoint with image."""
    print("Testing /v1/chat/completions (with image)...")

    # Try to load a test image
    test_image_path = "/home/jkuhm/src/MoltsBot-Source/medgemma-clone/medgemma/python/serving/testdata/image.jpeg"

    try:
        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Please describe it."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }
        response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Response: {result['choices'][0]['message']['content']}")
            print(f"Tokens used: {result['usage']}")
    except FileNotFoundError:
        print(f"Test image not found at {test_image_path}")
        print("Skipping image test...")
    print()


def test_multiturn_conversation():
    """Test multi-turn conversation."""
    print("Testing multi-turn conversation...")
    payload = {
        "messages": [
            {"role": "user", "content": "What is CT scan?"},
            {"role": "assistant", "content": "A CT scan (Computed Tomography) is a medical imaging technique that uses X-rays to create detailed cross-sectional images of the body."},
            {"role": "user", "content": "How is it different from MRI?"}
        ],
        "max_tokens": 200,
        "temperature": 0.7
    }
    response = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']}")
    print()


if __name__ == "__main__":
    print("="*60)
    print("MedGemma Server Test Suite")
    print("="*60)
    print()

    try:
        # Run tests
        test_health()
        test_simple_predict()
        test_chat_completion_text()
        test_chat_completion_with_image()
        test_multiturn_conversation()

        print("="*60)
        print("All tests completed!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server.")
        print("Please make sure the server is running on", SERVER_URL)
        print("Run: python simple_serving.py")
    except Exception as e:
        print(f"Error: {e}")
