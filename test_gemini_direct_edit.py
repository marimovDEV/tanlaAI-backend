"""
🚀 Gemini 3 Pro Image — rate limit bilan qayta urinish
"""
import time
import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

API_KEY = os.environ.get("GEMINI_API_KEY")
ROOM = "/Users/ogabek/Documents/projects/tanlaAI/hona.png"
DOOR = "/Users/ogabek/Documents/projects/tanlaAI/eshik.png"
RESULT = "/Users/ogabek/Documents/projects/tanlaAI/backend/gemini_direct_result.png"

# These 2 models exist and support image gen (hit rate limit = they work)
MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
]

client = genai.Client(api_key=API_KEY)

with open(ROOM, "rb") as f:
    room_bytes = f.read()
with open(DOOR, "rb") as f:
    door_bytes = f.read()

print(f"🖼 Xona: {Image.open(BytesIO(room_bytes)).size}")
print(f"🚪 Eshik: {Image.open(BytesIO(door_bytes)).size}")

prompt = (
    "First image is a room photo. Second image is a new door design.\n\n"
    "TASK: Replace the existing door in the room with the new door from the second image.\n\n"
    "RULES:\n"
    "- Place the new door naturally where the current door is\n"
    "- Keep the rest of the room unchanged (walls, floor, furniture, curtains)\n"
    "- Match lighting, perspective, and shadows to the room\n"
    "- The door should look naturally installed in the wall\n"
    "- Return ONLY the edited room image"
)

MAX_RETRIES = 5
WAIT_SECONDS = 15

for model in MODELS:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n🤖 {model} — urinish {attempt}/{MAX_RETRIES}")
            
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=room_bytes, mime_type="image/png"),
                    types.Part.from_bytes(data=door_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        img = Image.open(BytesIO(part.inline_data.data))
                        img.save(RESULT, format="PNG")
                        print(f"\n🎉 TAYYOR! {model}")
                        print(f"   📁 {RESULT}")
                        print(f"   📐 {img.size}")
                        exit(0)
                    elif part.text:
                        print(f"   💬 {part.text[:200]}")
            
            print(f"   ⚠️ Rasm kelmadi")
            
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"   ⏳ Rate limit — {WAIT_SECONDS}s kutmoqda...")
                time.sleep(WAIT_SECONDS)
            else:
                print(f"   ❌ {err[:200]}")
                break

print("\n💀 Ishlamadi. Yangi API key kerak: https://aistudio.google.com/apikey")
