import os
import io
import base64
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
from google import genai
from google.genai import types

def get_gemini_client():
    """
    Initializes the Vertex AI client. 
    Ready for direct key injection to bypass Windows credential loading issues.
    """
    import json
    import re
    from google.oauth2 import service_account

    project = getattr(settings, 'VERTEX_AI_PROJECT', 'ai-image-editor-492616')
    location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')
    key_path = settings.GOOGLE_APPLICATION_CREDENTIALS
    
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            
        # Нормализация PEM
        pk = info['private_key']
        match = re.search(r'-----BEGIN PRIVATE KEY-----(.*)-----END PRIVATE KEY-----', pk, re.DOTALL)
        if match:
            body = "".join(re.findall(r'[A-Za-z0-9+/=]', match.group(1)))
            formatted_body = "\n".join(body[i:i+64] for i in range(0, len(body), 64))
            info['private_key'] = f"-----BEGIN PRIVATE KEY-----\n{formatted_body}\n-----END PRIVATE KEY-----\n"
        
        credentials = service_account.Credentials.from_service_account_info(info)
    except Exception as e:
        print(f"DEBUG: Falling back from file loading due to: {e}")
        # Если файл совсем не грузится, можно будет захардкодить ключ временно для теста
        raise e
    
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials
    )

def process_product_with_ai(product):
    """
    Uses Photoroom API to isolate the door and remove the background.
    Results in a high-quality transparent PNG.
    """
    import requests
    from django.conf import settings
    
    try:
        if product.ai_status != 'none':
            return
            
        print(f"DEBUG: [Photoroom] Processing creation for Product {product.id}...")
        product.ai_status = 'processing'
        product.save(update_fields=['ai_status'])
        
        # Ensure original image is saved
        if not product.original_image:
            image_name = os.path.basename(product.image.name)
            original_content = product.image.read()
            product.original_image.save(image_name, ContentFile(original_content), save=False)
        product.save()

        api_key = getattr(settings, 'PHOTOROOM_API_KEY', os.getenv('PHOTOROOM_API_KEY'))
        if not api_key:
            raise ValueError("Photoroom API key missing in settings.")
            
        url = "https://sdk.photoroom.com/v1/segment"
        with open(product.original_image.path, "rb") as img_file:
            files = {"image_file": img_file}
            headers = {"x-api-key": api_key}
            response = requests.post(url, files=files, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Photoroom API failed: {response.status_code} - {response.text}")
            
        # Success: Save the transparent PNG
        content = response.content
        product.image.save(f"processed_{product.id}.png", ContentFile(content), save=False)
        product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(content), save=False)
        
        product.ai_status = 'completed'
        product.save(update_fields=['image', 'image_no_bg', 'ai_status'])
        print(f"DEBUG: [Photoroom] Success! Background removed for product {product.id}")
        return True
        
    except Exception as e:
        print(f"ERROR: [Photoroom] Exception occurred: {e}")
        product.ai_status = 'error'
        product.save(update_fields=['ai_status'])
        return False

    except Exception as e:
        print(f"DEBUG: [Gemini AI] Error during creation: {e}")
        product.ai_status = 'error'
        product.save(update_fields=['ai_status'])

def create_binary_mask(width: int, height: int, box_1000: list[int] = None, polygon: list[list[int]] = None, invert: bool = False) -> io.BytesIO:
    """
    Creates a black and white mask image.
    Supports either a bounding box [ymin, xmin, ymax, xmax] OR a list of [x, y] polygon points.
    If invert is True, the identified object becomes black (0) and background becomes white (255).
    """
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create black image (0)
    bg_color = 255 if invert else 0
    fg_color = 0 if invert else 255
    
    mask = Image.new("L", (width, height), bg_color)
    draw = ImageDraw.Draw(mask)
    
    if polygon:
        # Draw explicit polygon from GPT-4o points
        points = [(int(p[0]), int(p[1])) for p in polygon]
        draw.polygon(points, fill=fg_color)
    elif box_1000:
        # Map coordinates with a buffer to ensure zero clipping
        ymin, xmin, ymax, xmax = box_1000
        
        left = int(xmin * width / 1000)
        top = int(ymin * height / 1000)
        right = int(xmax * width / 1000)
        bottom = int(ymax * height / 1000)
        draw.rectangle([left, top, right, bottom], fill=fg_color)
    
    buf = io.BytesIO()
    mask.save(buf, format='PNG')
    buf.seek(0)
    return buf

def replace_background_with_green(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """
    Uses Gemini (Nano Banana) to replace the background with a solid #00FF00 green color.
    """
    client = get_gemini_client()
    
    config = types.EditImageConfig(
        edit_mode='INPAINT_EDIT',
        number_of_images=1,
        output_mime_type='image/png',
        mask=types.Image(image_bytes=mask_bytes)
    )
    
    print("DEBUG: [Gemini/Nano Banana] Replacing background with Green Screen (#00FF00)...")
    response = client.models.edit_image(
        model='imagen-3.0-capability-001',
        prompt=(
            "Keep the main door object exactly as it is, preserving all its dark wooden textures, frame details, and glass panels. "
            "The background behind and around the door must be replaced with a solid, flat, bright green color (#00FF00). "
            "Do not alter the door's structure or colors."
        ),
        reference_images=[
            types.RawReferenceImage(
                reference_image=types.Image(image_bytes=image_bytes),
                reference_id=0
            )
        ],
        config=config
    )
    
    if not response.generated_images:
        raise ValueError("Gemini failed to generate green-screen image.")
        
    return response.generated_images[0].image.image_bytes

    return response.generated_images[0].image.image_bytes

def analyze_product_details(product_img) -> str:
    """
    Uses GPT-4o Vision to describe the door product in detail.
    This description guides Imagen's holistic generation.
    """
    import io
    import base64
    import requests
    import json
    from django.conf import settings
    from PIL import Image

    print("DEBUG: [GPT-4o Vision] Analyzing product details for semantic guide...")
    
    # Resize for faster analysis
    p_img = product_img.convert("RGB")
    p_img.thumbnail((512, 1024))
    r_bytes = io.BytesIO()
    p_img.save(r_bytes, format='JPEG', quality=85)
    base64_image = base64.b64encode(r_bytes.getvalue()).decode('utf-8')
    
    openai_key = getattr(settings, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        return "A high-quality premium door matching the reference style."

    prompt = (
        "Describe this door in 1-2 concise sentences for a professional interior designer.\n"
        "Focus on: color, material (wood, metal, etc), number of panels, glass style, and hardware (handle) color/finish.\n"
        "Example: 'A classic white double door with 8 rectangular glass panels and polished gold handles.'"
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        desc = response.json()['choices'][0]['message']['content']
        print(f"DEBUG: [GPT-4o Vision] Product description: {desc}")
        return desc
    except Exception as e:
        print(f"ERROR: [GPT-4o Vision] Product analysis failed: {e}")
        return "A premium door with decorative details and high-end finish."

def analyze_room_for_placement(room_img) -> dict:
    """
    Uses GPT-4o Vision to find the best wall spot for a door installation.
    Returns exact coordinates and scene context.
    """
    import io
    import base64
    import requests
    import json
    from django.conf import settings

    print("DEBUG: [GPT-4o Vision] Analyzing room for optimal door placement...")
    
    # Resize for faster/cheaper analysis if needed
    r_bytes = io.BytesIO()
    room_img.save(r_bytes, format='JPEG', quality=85)
    base64_image = base64.b64encode(r_bytes.getvalue()).decode('utf-8')
    
    openai_key = getattr(settings, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
    if not openai_key:
        raise ValueError("OpenAI API Key missing")

    prompt = (
        "Analyze this room for a premium door installation.\n"
        "1. Identify the single best vertical wall area for a new door (normalized JSON).\n"
        "2. Detect the floor line (y-coordinate).\n"
        "3. Identify any clutter, wires, or cables near the doorway to be removed.\n"
        "4. Determine the room's interior style (e.g., 'Classic Uzbek', 'Modern Minimalist').\n"
        "5. Suggest a matching lighting profile ('cinematic soft', 'bright day', 'warm interior').\n\n"
        "Return accurate JSON:\n"
        "{\n"
        "  \"door_box\": {\"ymin\": 0.2, \"xmin\": 0.4, \"ymax\": 0.8, \"xmax\": 0.55},\n"
        "  \"floor_y\": 0.8,\n"
        "  \"clutter_removal\": true,\n"
        "  \"style\": \"Premium Classic\",\n"
        "  \"lighting\": \"warm cinematic\",\n"
        "  \"perspective_angle\": \"straight/left/right\"\n"
        "}"
    )
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        result = json.loads(response.json()['choices'][0]['message']['content'])
        print(f"DEBUG: [GPT-4o Vision] Analysis complete: {result}")
        return result
    except Exception as e:
        print(f"ERROR: [GPT-4o Vision] Analysis failed: {e}")
        return {
            "door_box": {"ymin": 0.2, "xmin": 0.4, "ymax": 0.85, "xmax": 0.6},
            "floor_y": 0.85,
            "lighting": "natural",
            "perspective_angle": "straight"
        }

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None):
    """
    v27 — The Master Builder.
    BALANCED REPLACEMENT: Strong Removal + Strong Insertion.
    Forces the specific subject design into the surgical opening.
    """
    import time
    start_t = time.time()
    
    def LOG(step, msg):
        elapsed = time.time() - start_t
        print(f"[v27 {elapsed:6.2f}s] STEP {step}: {msg}")
    
    LOG(0, f"=== THE MASTER BUILDER (v27) ===")
    
    try:
        from PIL import Image, ImageOps, ImageDraw, ImageFilter
        from google.genai import types
        import numpy as np
        import io
        import os
        
        # ====== STEP 1: AI Client ======
        from shop.services import AIService
        client = AIService.get_gemini_client()
        
        # ====== STEP 2: Resource Loading (1024 Resolution) ======
        room_img_raw = ImageOps.exif_transpose(Image.open(room_image_path)).convert("RGB")
        room_img = room_img_raw.resize((1024, 1024), Image.LANCZOS)
        rw, rh = room_img.size
        
        # Use ORIGINAL image for subject reference (cleanest design)
        door_field = product.original_image or product.image
        door_asset = ImageOps.exif_transpose(Image.open(door_field.path)).convert("RGB")
        door_asset.thumbnail((1024, 1024))
        LOG(2, f"Loaded Global Resources: {os.path.basename(door_field.path)}")
        
        # ====== STEP 3: Surgical Target Analysis ======
        try:
            if not box_1000:
                room_analysis = analyze_room_for_placement(room_img)
                bx = room_analysis['door_box']
                box_1000 = [int(bx['ymin']*1000), int(bx['xmin']*1000), int(bx['ymax']*1000), int(bx['xmax']*1000)]
            else:
                room_analysis = {"lighting": "natural", "style": "Premium"}
        except Exception:
            box_1000 = box_1000 or [200, 350, 950, 650]
            room_analysis = {"lighting": "natural", "style": "Premium"}

        # Extract semantic details to 'anchor' the AI's understanding
        product_desc = analyze_product_details(door_asset)
        
        ymin, xmin, ymax, xmax = box_1000
        left, top, right, bottom = int(xmin*rw/1000), int(ymin*rh/1000), int(xmax*rw/1000), int(ymax*rh/1000)
        
        # MASK: Solid core with minimal 5px blur for surgical insertion
        mask_img = Image.new("L", (1024, 1024), 0)
        draw = ImageDraw.Draw(mask_img)
        # Wider mask to capture the entire frame area
        draw.rectangle([left-25, top-25, right+25, bottom+25], fill=255)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(5))
        
        room_buf = io.BytesIO()
        room_img.save(room_buf, format='PNG')
        
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format='PNG')
        
        door_buf = io.BytesIO()
        door_asset.save(door_buf, format='PNG')
        
        # ====== STEP 4: Forceful Insertion Prompt ======
        prompt = (
            f"HIGH-FIDELITY ARCHITECTURAL INSTALLATION.\n"
            f"Reference 1: Interior wall with an entry opening.\n"
            f"Reference 2: Specific custom door product (Subject).\n"
            f"GOAL:\n"
            f"Install the EXACT DOOR from Reference 2 into the opening in Reference 1.\n"
            f"MANDATORY DESIGN: {product_desc}. Replicate the precise colors, geometric patterns, and hardware.\n"
            f"CONFIGURATION: The door MUST be CLOSED. Do not show the interior of the doorway.\n"
            f"INTEGRATION: Connect the door frame to the Reference 1 walls seamlessly. "
            f"Match lighting and shadows of the room. Photorealistic 8k quality."
        )

        LOG(5, "Invoking Imagen 3 Master Builder...")
        response = None
        try:
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=prompt,
                reference_images=[
                    types.RawReferenceImage(reference_id=1, reference_image=types.Image(image_bytes=room_buf.getvalue())),
                    types.SubjectReferenceImage(
                        reference_id=2, 
                        reference_image=types.Image(image_bytes=door_buf.getvalue()),
                        config=types.SubjectReferenceConfig(subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT')
                    ),
                    types.MaskReferenceImage(
                        reference_id=3, 
                        reference_image=types.Image(image_bytes=mask_buf.getvalue()),
                        config=types.MaskReferenceConfig(mask_mode='MASK_MODE_USER_PROVIDED')
                    ),
                ],
                config=types.EditImageConfig(
                    edit_mode='EDIT_MODE_INPAINT_EDIT',
                    number_of_images=1,
                    output_mime_type='image/png',
                ),
            )
        except Exception as ai_err:
            LOG("!", f"Imagen call failed: {ai_err}")
            response = None

        if response and response.generated_images:
            gen_img_bytes = response.generated_images[0].image.image_bytes
            final_img = Image.open(io.BytesIO(gen_img_bytes)).resize(room_img_raw.size)
            final_img.save(result_image_path, format='JPEG', quality=95)
            LOG(7, f"SUCCESS: v27 result saved: {result_image_path}")
            return result_image_path
        else:
            LOG(7, "AI Failed. Returning fallback.")
            room_img_raw.save(result_image_path, format='JPEG', quality=90)
            return result_image_path

    except Exception as e:
        LOG("X", f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

    except Exception as e:
        LOG("X", f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

    except Exception as e:
        LOG("X", f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e

    except Exception as e:
        LOG("X", f"💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise e
