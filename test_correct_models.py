from google import genai
from google.genai import types
from PIL import Image
import io

client = genai.Client(api_key="AIzaSyARA_dDh__hqSUp6jUF-J9dJtpsFDoJ7cw")

room = Image.new('RGB', (512, 512), color=(200, 180, 160))
buf = io.BytesIO()
room.save(buf, format='PNG')
room_bytes = buf.getvalue()

door = Image.new('RGB', (200, 400), color=(240, 230, 220))
buf2 = io.BytesIO()
door.save(buf2, format='PNG')
door_bytes = buf2.getvalue()

models_to_try = [
    'gemini-2.5-flash-image',
    'gemini-3.1-flash-image-preview',
    'gemini-3-pro-image-preview',
    'gemini-2.0-flash',
]

for model_name in models_to_try:
    try:
        print(f"Trying: {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                "Replace the door in the first image with the door from the second image. Keep the room unchanged.",
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    print(f"  SUCCESS with {model_name}! Got image back.")
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    img.save(f"test_result_{model_name.replace('.','_').replace('-','_')}.png")
                    break
                elif part.text:
                    print(f"  Got text: {part.text[:150]}")
        else:
            print(f"  Empty response from {model_name}")
    except Exception as e:
        err = str(e)[:200]
        print(f"  FAILED {model_name}: {err}")
