import os, sys
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from google import genai
from google.genai import types
from PIL import Image
import io

keys = [
    "AIzaSyARA_dDh__hqSUp6jUF-J9dJtpsFDoJ7cw", # Key 1
    "AIzaSyDwo0aAUA8Yv3oac3YFvVteTe_2pTrSKyQ"  # Key 2
]

room = Image.new('RGB', (512, 512), color=(200, 180, 160))
buf = io.BytesIO()
room.save(buf, format='PNG')
room_bytes = buf.getvalue()

door = Image.new('RGB', (200, 400), color=(240, 230, 220))
buf2 = io.BytesIO()
door.save(buf2, format='PNG')
door_bytes = buf2.getvalue()

models = [
    'gemini-3.1-flash-image-preview',
    'gemini-3-pro-image-preview',
    'gemini-2.5-flash-image',
]

for i, key in enumerate(keys):
    print(f"\n--- Testing Key {i+1} ({key[:10]}...) ---")
    client = genai.Client(api_key=key)
    for model in models:
        try:
            print(f"Trying {model}...")
            res = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                    types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                    "Replace door."
                ],
                config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
            )
            print(f"  SUCCESS! Model {model} is working.")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                print(f"  FAILED: Quota exhausted (429)")
            elif "404" in msg or "not found" in msg.lower():
                print(f"  FAILED: Model not found (404)")
            elif "400" in msg:
                print(f"  FAILED: Invalid argument (400) - check modalities support")
            else:
                print(f"  FAILED: {msg[:150]}")

