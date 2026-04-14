"""
Test script to verify which Gemini models support image generation.
Run: python3 scripts/test_gemini_image_gen.py
"""
import os
import sys

# Minimal test - no Django needed
from google import genai
from google.genai import types

API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyARA_dDh__hqSUp6jUF-J9dJtpsFDoJ7cw')

print(f"=== Gemini Image Generation Test ===")
print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")

client = genai.Client(api_key=API_KEY)

# Step 1: List models that might support image generation
print("\n--- Step 1: Listing available models ---")
try:
    for model in client.models.list():
        name = model.name
        # Only show image-related or flash models
        if any(k in name.lower() for k in ['flash', 'image', 'imagen', 'gemini-2', 'gemini-3']):
            print(f"  ✓ {name}")
except Exception as e:
    print(f"  ✗ Could not list models: {e}")

# Step 2: Try generate_content with image output on different models
models_to_test = [
    'gemini-2.0-flash-exp',
    'gemini-2.0-flash',
    'gemini-2.0-flash-preview-image-generation',
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-flash',
    'gemini-2.5-flash-image',
]

print("\n--- Step 2: Testing generate_content with IMAGE modality ---")
for model_name in models_to_test:
    try:
        print(f"\n  Testing: {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents="Generate a simple red circle on white background",
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            has_image = False
            has_text = False
            for part in parts:
                if part.inline_data is not None:
                    has_image = True
                    print(f"    ✓ IMAGE returned! Size: {len(part.inline_data.data)} bytes, MIME: {part.inline_data.mime_type}")
                if part.text:
                    has_text = True
                    print(f"    ✓ TEXT: {part.text[:100]}")
            
            if not has_image and not has_text:
                print(f"    ✗ Empty parts in response")
        else:
            print(f"    ✗ No candidates/content in response")
            
    except Exception as e:
        print(f"    ✗ FAILED: {e}")

print("\n=== Test Complete ===")
