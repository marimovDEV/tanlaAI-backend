"""
🚀 Vertex AI Direct Edit Test (Gemini 3.1 Flash Image)
"""
import os
import json
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Config
KEY_PATH = "/Users/ogabek/Documents/projects/tanlaAI/backend/google-cloud-key.json"
ROOM = "/Users/ogabek/Documents/projects/tanlaAI/hona.png"
DOOR = "/Users/ogabek/Documents/projects/tanlaAI/eshik.png"
RESULT = "/Users/ogabek/Documents/projects/tanlaAI/backend/vertex_direct_result.png"

def test_vertex_direct():
    if not os.path.exists(KEY_PATH):
        print(f"❌ Key path not found: {KEY_PATH}")
        return

    with open(KEY_PATH, 'r') as f:
        info = json.load(f)
    
    client = genai.Client(
        vertexai=True,
        project=info['project_id'],
        location="us-central1"
    )

    with open(ROOM, "rb") as f:
        room_bytes = f.read()
    with open(DOOR, "rb") as f:
        door_bytes = f.read()

    # The model ID from the list
    model_name = "gemini-3.1-flash-image-preview"

    prompt = (
        "Birinchi rasm — xona fotosuratidir. Ikkinchi rasm — yangi eshik dizaynidir.\n\n"
        "VAZIFA: Xonadagi mavjud eshikni ikkinchi rasmdagi yangi eshik bilan almashtir.\n\n"
        "QOIDALAR:\n"
        "- Yangi eshikni xonadagi eshik o'rniga tabiiy ko'rinishda joylashtir\n"
        "- Xonaning qolgan qismini (devor, pol, mebel, parda) o'zgartirma\n"
        "- Yoritish, perspektiva va soyalarni xonaga mos qil\n"
        "- Eshik devorga tabiiy o'rnatilgandek ko'rinsin\n"
        "- Faqat bitta tahrirlangan xona rasmini qaytar"
    )

    print(f"🤖 Testing model: {model_name} on Vertex AI...")
    try:
        response = client.models.generate_content(
            model=model_name,
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
                if part.inline_data:
                    img = Image.open(BytesIO(part.inline_data.data))
                    img.save(RESULT, format="PNG")
                    print(f"✅ SUCCESS! Result saved to {RESULT}")
                    return True
                elif part.text:
                    print(f"💬 Text response: {part.text[:200]}")
        
        print("⚠️ No image in response")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

if __name__ == "__main__":
    test_vertex_direct()
