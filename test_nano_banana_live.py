"""Nano Banana — 2-Step AI (Detector + Editor)"""
import sys
import os
import json
import re
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from io import BytesIO

# API Key fallback for testing
API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMINI_API_KEY")
ROOM = "/Users/ogabek/Documents/projects/tanlaAI/hona.png"
DOOR = "/Users/ogabek/Documents/projects/tanlaAI/eshik.png"
RESULT = "/Users/ogabek/Documents/projects/tanlaAI/backend/nano_banana_result.png"

# Client Initialization - Try API key first, then Vertex AI
def get_client():
    # Try API key first (simpler, no project restrictions)
    if API_KEY:
        try:
            c = genai.Client(api_key=API_KEY)
            c.models.get(model='gemini-2.0-flash')
            print("✅ Using API key client")
            return c
        except Exception as e:
            print(f"⚠️ API key failed: {str(e)[:100]}")
    
    # Try Vertex AI
    key_path = "/Users/ogabek/Documents/projects/tanlaAI/backend/google-cloud-key.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    
    for project_id in ["ai-image-editor-492616", "project-b79a8b66-7b7a-4251-b56"]:
        try:
            c = genai.Client(vertexai=True, project=project_id, location="us-central1")
            print(f"✅ Using Vertex AI client (project={project_id})")
            return c
        except Exception as e:
            print(f"⚠️ Vertex {project_id} failed: {str(e)[:100]}")
    
    raise RuntimeError("No working client found!")

client = get_client()

# Models to try (ordered by preference)
DETECTION_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash',
    'gemini-2.5-pro',
]
INPAINT_MODELS = [
    'gemini-2.5-flash-image',
    'gemini-2.5-flash',
    'gemini-2.0-flash',
]

def detect_door_location(room_bytes, w_img, h_img):
    """Phase 1: AI detects where the door should go."""
    detection_prompt = (
        "Analyze this room image. I want to install a new door.\n"
        "Return ONLY a JSON object with the bounding box for the door placement.\n"
        "Format: {\"ymin\": 0-1000, \"xmin\": 0-1000, \"ymax\": 0-1000, \"xmax\": 0-1000}\n"
        "Use normalized coordinates (0 to 1000).\n"
        "The door should fit naturally on the wall."
    )
    
    for model in DETECTION_MODELS:
        try:
            print(f"  🤖 Trying {model} for detection...")
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                    detection_prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                )
            )
            
            text = response.text.strip()
            # Try to extract JSON if wrapped in markdown
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                text = json_match.group(0)
            
            coords = json.loads(text)
            ymin = int(coords.get('ymin', 200))
            xmin = int(coords.get('xmin', 300))
            ymax = int(coords.get('ymax', 800))
            xmax = int(coords.get('xmax', 700))
            
            # Convert normalized to pixels
            py_ymin = int(ymin * h_img / 1000)
            py_xmin = int(xmin * w_img / 1000)
            py_ymax = int(ymax * h_img / 1000)
            py_xmax = int(xmax * w_img / 1000)
            
            print(f"  ✅ Detection success with {model}: ({py_xmin},{py_ymin})-({py_xmax},{py_ymax})")
            return py_xmin, py_ymin, py_xmax, py_ymax, model
        except Exception as e:
            print(f"  ❌ {model}: {str(e)[:200]}")
    
    # Fallback: center of image
    print("  ⚠️ All detections failed, using center fallback")
    fx = int(w_img * 0.25)
    fy = int(h_img * 0.15)
    fw = int(w_img * 0.5)
    fh = int(h_img * 0.75)
    return fx, fy, fx + fw, fy + fh, 'fallback-center'


def inpaint_door(room_bytes, mask_bytes, door_bytes, edit_prompt):
    """Phase 2: AI places the door using the mask."""
    for model in INPAINT_MODELS:
        try:
            print(f"  🤖 Trying {model} for inpainting...")
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                    types.Part.from_bytes(data=mask_bytes, mime_type='image/png'),
                    types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                    edit_prompt,
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                )
            )
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img = Image.open(BytesIO(part.inline_data.data))
                        print(f"  ✅ Inpaint success with {model}: {img.size}")
                        return img, model
                    elif part.text:
                        print(f"  💬 {part.text[:150]}")
            print(f"  ⚠️ {model}: no image in response")
        except Exception as e:
            print(f"  ❌ {model}: {str(e)[:200]}")
    
    return None, None


def test_nano_banana():
    print("🍌 NANO BANANA AUTO-EDIT TEST\n")
    
    if not os.path.exists(ROOM):
        print(f"❌ Xona rasmi topilmadi: {ROOM}")
        return
    if not os.path.exists(DOOR):
        print(f"❌ Eshik rasmi topilmadi: {DOOR}")
        return

    # Load Room
    with open(ROOM, 'rb') as f:
        room_bytes = f.read()
    room_img = Image.open(BytesIO(room_bytes))
    w_img, h_img = room_img.size
    print(f"🖼 Room: {w_img}x{h_img}")
    
    # ═══════════════════════════════════════
    # Phase 1: AI Detection
    # ═══════════════════════════════════════
    print("\n🔍 1-BOSQICH: Devorni analiz qilyapman...")
    xmin, ymin, xmax, ymax, det_model = detect_door_location(room_bytes, w_img, h_img)
    print(f"📍 Box: ({xmin},{ymin})-({xmax},{ymax}) via {det_model}")

    # ═══════════════════════════════════════
    # Phase 2: Auto Masking
    # ═══════════════════════════════════════
    print("\n🎨 2-BOSQICH: Maskani chizyapman...")
    mask = Image.new('L', (w_img, h_img), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
    
    mask_buf = BytesIO()
    mask.save(mask_buf, format='PNG')
    mask_bytes = mask_buf.getvalue()
    print(f"✅ Mask created: {xmax-xmin}x{ymax-ymin} pixels")

    # ═══════════════════════════════════════
    # Phase 3: Inpainting
    # ═══════════════════════════════════════
    print("\n🚪 3-BOSQICH: Eshikni joylashtiryapman...")
    with open(DOOR, 'rb') as f:
        door_bytes = f.read()
        
    edit_prompt = (
        "Image 1: Original room photo.\n"
        "Image 2: MASK — white area shows where to place the new door.\n"
        "Image 3: The NEW DOOR design to install.\n\n"
        "TASK: Replace the masked area with the new door.\n"
        "RULES:\n"
        "- Match the room's lighting and perspective\n"
        "- Blend edges perfectly (shadows, lighting)\n"
        "- Do NOT change anything outside the mask\n"
        "- Make the door look naturally installed in the wall\n"
        "- Return ONLY the edited room image"
    )
    
    result_img, inp_model = inpaint_door(room_bytes, mask_bytes, door_bytes, edit_prompt)
    
    if result_img:
        result_img.save(RESULT, format='PNG')
        print(f"\n🎉 TAYYOR! '{RESULT}' saqlandi.")
        print(f"   Detection: {det_model}")
        print(f"   Inpaint:   {inp_model}")
        print(f"   Size:      {result_img.size}")
    else:
        print("\n💀 Barcha modellar muvaffaqiyatsiz.")


if __name__ == "__main__":
    test_nano_banana()
