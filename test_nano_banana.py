from google import genai
from google.genai import types
from PIL import Image
import io
import os

# Use the new key provided by the user
api_key = "AIzaSyDwo0aAUA8Yv3oac3YFvVteTe_2pTrSKyQ"
client = genai.Client(api_key=api_key)

room = Image.new('RGB', (1024, 1024), color=(200, 180, 160))
buf = io.BytesIO()
room.save(buf, format='PNG')
room_bytes = buf.getvalue()

door = Image.new('RGB', (400, 800), color=(240, 230, 220))
buf2 = io.BytesIO()
door.save(buf2, format='PNG')
door_bytes = buf2.getvalue()

# Nano Banana Pro = gemini-3-pro-image-preview
# Nano Banana 2 = gemini-3.1-flash-image-preview
models_to_try = [
    'gemini-3-pro-image-preview',
    'gemini-3.1-flash-image-preview',
    'gemini-2.5-flash-image',
]

for model_name in models_to_try:
    try:
        print(f"Testing Nano Banana Tier: {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                "Professional interior design: Replace the existing door with the new door asset. Maintain 100% lighting consistency and wall texture integrity.",
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    print(f"  SUCCESS! {model_name} (Nano Banana) generated the image.")
                    exit(0)
    except Exception as e:
        print(f"  Error with {model_name}: {str(e)[:100]}")

