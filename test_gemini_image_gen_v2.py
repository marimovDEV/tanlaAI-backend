import os, sys
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

import django
django.setup()

from google import genai
from google.genai import types
from PIL import Image
import io

# Test with API key
api_key = "AIzaSyARA_dDh__hqSUp6jUF-J9dJtpsFDoJ7cw"
client = genai.Client(api_key=api_key)

# Create dummy images
room = Image.new('RGB', (512, 512), color=(200, 180, 160))
buf = io.BytesIO()
room.save(buf, format='PNG')
room_bytes = buf.getvalue()

door = Image.new('RGB', (200, 400), color=(240, 230, 220))
buf2 = io.BytesIO()
door.save(buf2, format='PNG')
door_bytes = buf2.getvalue()

models_to_try = [
    'gemini-2.0-flash-exp',
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.0-flash-preview-image-generation',
]

for model_name in models_to_try:
    try:
        print(f"Trying: {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                "Replace the door in the first image with the door from the second image.",
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    print(f"  SUCCESS with {model_name}! Got image back.")
                    break
                elif part.text:
                    print(f"  Got text response: {part.text[:100]}")
            else:
                print(f"  No image in response from {model_name}")
        else:
            print(f"  Empty response from {model_name}")
    except Exception as e:
        print(f"  FAILED {model_name}: {e}")

