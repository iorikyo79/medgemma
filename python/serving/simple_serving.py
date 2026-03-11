#!/usr/bin/env python3
"""Simple MedGemma serving script using transformers directly."""

import base64
import io
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# Configuration
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = str(SCRIPT_DIR / "models" / "medgemma-4b-it")
HOST = "0.0.0.0"
PORT = 8080

print("Initializing MedGemma 4B server...")
print(f"Model path: {MODEL_PATH}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Initialize processor
print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Initialize model
print("Loading model with transformers...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully!")

app = Flask(__name__)


def encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def process_image_for_model(image: Image.Image) -> str:
    """Process image and return base64 encoded string for the model."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if too large (optional optimization)
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Encode to base64 JPEG
    return encode_image_to_base64(image, format="JPEG")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "medgemma-4b-it",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.get_json()

        # Extract messages and parameters
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 500)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        # Convert messages to prompt using processor
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare inputs
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_text = processor.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Count tokens (approximate)
        prompt_tokens = inputs['input_ids'].shape[1]
        completion_tokens = outputs[0].shape[0] - prompt_tokens

        response = {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "medgemma-4b-it",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Simple predict endpoint for basic testing."""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Prepare inputs
        inputs = processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated_text = processor.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return jsonify({
            "response": generated_text,
            "prompt": prompt
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("MedGemma 4B Server Ready!")
    print(f"{'='*60}")
    print(f"Server running on http://{HOST}:{PORT}")
    print(f"\nAvailable endpoints:")
    print(f"  - GET  http://localhost:{PORT}/health")
    print(f"  - POST http://localhost:{PORT}/v1/chat/completions")
    print(f"  - POST http://localhost:{PORT}/predict")
    print(f"{'='*60}\n")

    app.run(host=HOST, port=PORT, debug=False)
