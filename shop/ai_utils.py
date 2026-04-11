import os
import io
import base64
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from django.core.files.base import ContentFile
from django.conf import settings
from google import genai
from google.genai import types

def get_gemini_client():
    import json
    import re
    from google.oauth2 import service_account
    project = getattr(settings, 'VERTEX_AI_PROJECT', 'ai-image-editor-492616')
    location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')
    key_path = settings.GOOGLE_APPLICATION_CREDENTIALS
    try:
        with open(key_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        credentials = service_account.Credentials.from_service_account_info(info)
    except Exception as e:
        print(f"DEBUG: Client Loading Error: {e}")
        raise e
    return genai.Client(vertexai=True, project=project, location=location, credentials=credentials)

def process_product_with_ai(product):
    import requests
    try:
        if product.ai_status != 'none': return
        product.ai_status = 'processing'; product.save(update_fields=['ai_status'])
        if not product.original_image:
            product.original_image.save(os.path.basename(product.image.name), ContentFile(product.image.read()), save=False)
        product.save()
        api_key = getattr(settings, 'PHOTOROOM_API_KEY', os.getenv('PHOTOROOM_API_KEY'))
        url = "https://sdk.photoroom.com/v1/segment"
        with open(product.original_image.path, "rb") as img_file:
            response = requests.post(url, files={"image_file": img_file}, headers={"x-api-key": api_key})
        if response.status_code == 200:
            content = response.content
            product.image.save(f"processed_{product.id}.png", ContentFile(content), save=False)
            product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(content), save=False)
            product.ai_status = 'completed'
        else: product.ai_status = 'error'
        product.save(update_fields=['image', 'image_no_bg', 'ai_status'])
    except Exception:
        product.ai_status = 'error'; product.save(update_fields=['ai_status'])

def analyze_product_details(product_img) -> str:
    try:
        import base64, requests, json
        r_bytes = io.BytesIO(); product_img.save(r_bytes, format='JPEG', quality=85)
        b64 = base64.b64encode(r_bytes.getvalue()).decode('utf-8')
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe this door product briefly (max 15 words)."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
        return resp.json()['choices'][0]['message']['content'].strip()
    except Exception: return "a modern architectural door"

def analyze_room_for_placement(room_img) -> dict:
    try:
        import base64, requests, json
        r_bytes = io.BytesIO(); room_img.save(r_bytes, format='JPEG', quality=85)
        b64 = base64.b64encode(r_bytes.getvalue()).decode('utf-8')
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        p = "Analyze room JSON: {\"design_dna\": \"desc\", \"door_box\": {\"ymin\": 0.2, \"xmin\": 0.4, \"ymax\": 0.8, \"xmax\": 0.6}, \"lighting\": \"mood\"}"
        payload = {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": p}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], "response_format": {"type": "json_object"}}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        return json.loads(resp.json()['choices'][0]['message']['content'])
    except Exception:
        return {"design_dna": "Modern interior", "door_box": {"ymin": 0.2, "xmin": 0.4, "ymax": 0.8, "xmax": 0.6}, "lighting": "natural"}

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None):
    """ v32 - Unstoppable Pipeline """
    import time
    start_t = time.time()
    def LOG(step, msg): print(f"[v32 {time.time()-start_t:6.2f}s] {step}: {msg}")
    
    room_raw = None
    try:
        room_raw = ImageOps.exif_transpose(Image.open(room_image_path)).convert("RGB")
        room_img = room_raw.resize((1024, 1024), Image.LANCZOS)
        door_path = (product.original_image or product.image).path
        door_img = ImageOps.exif_transpose(Image.open(door_path)).convert("RGB")
        door_img.thumbnail((1024, 1024))
        
        LOG(1, "Resources ready.")
        from shop.services import AIService
        client = None
        try: client = AIService.get_gemini_client()
        except: LOG("!", "AI Client failed.")

        room_dna = analyze_room_for_placement(room_img)
        prod_dna = analyze_product_details(door_img)
        
        r_buf = io.BytesIO(); room_img.save(r_buf, format='PNG')
        d_buf = io.BytesIO(); door_img.save(d_buf, format='PNG')
        
        if client:
            # STAGE 1: Full Reconstruction
            try:
                LOG(3, "Stage 1: Reconstruction...")
                p1 = f"ARCHITECTURAL RECONSTRUCTION. STYLE [0]: {room_dna['design_dna']}. SUBJECT [1]: {prod_dna}. Closed door. 8k."
                res1 = client.models.generate_images(
                    model='imagen-3.0-capability-001', prompt=p1,
                    config=types.GenerateImagesConfig(
                        number_of_images=1, output_mime_type='image/png',
                        style_reference_config=types.StyleReferenceConfig(style_reference_images=[types.StyleReferenceImage(reference_id=0, image=types.Image(image_bytes=r_buf.getvalue()))]),
                        subject_reference_config=types.SubjectReferenceConfig(subject_reference_images=[types.SubjectReferenceImage(reference_id=1, image=types.Image(image_bytes=d_buf.getvalue()), subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT')])
                    )
                )
                if res1 and res1.generated_images:
                    Image.open(io.BytesIO(res1.generated_images[0].image.image_bytes)).resize(room_raw.size).convert("RGB").save(result_image_path, format='JPEG', quality=95)
                    LOG(7, "STAGE 1 SUCCESS.")
                    return result_image_path
            except Exception as e1: LOG("!", f"Stage 1 failed: {e1}")

            # STAGE 2: Inpainting
            try:
                LOG(4, "Stage 2: Inpainting...")
                bx = room_dna['door_box']
                rw, rh = room_img.size
                l, t, r, b = int(bx['xmin']*rw), int(bx['ymin']*rh), int(bx['xmax']*rw), int(bx['ymax']*rh)
                mask = Image.new("L", (1024, 1024), 0)
                ImageDraw.Draw(mask).rectangle([l-20, t-20, r+20, b+20], fill=255)
                m_buf = io.BytesIO(); mask.filter(ImageFilter.GaussianBlur(5)).save(m_buf, format='PNG')
                
                p2 = f"INSERTION: Install [1] into [0]. DESIGN: {prod_dna}. Closed door. 8k."
                res2 = client.models.edit_image(
                    model='imagen-3.0-capability-001', prompt=p2,
                    reference_images=[
                        types.RawReferenceImage(reference_id=0, reference_image=types.Image(image_bytes=r_buf.getvalue())),
                        types.SubjectReferenceImage(reference_id=1, image=types.Image(image_bytes=d_buf.getvalue()), config=types.SubjectReferenceConfig(subject_type='SUBJECT_REFERENCE_TYPE_PRODUCT')),
                        types.MaskReferenceImage(reference_id=2, reference_image=types.Image(image_bytes=m_buf.getvalue()))
                    ],
                    config=types.EditImageConfig(edit_mode='EDIT_MODE_INPAINT_INSERTION', number_of_images=1, output_mime_type='image/png'),
                )
                if res2 and res2.generated_images:
                    Image.open(io.BytesIO(res2.generated_images[0].image.image_bytes)).resize(room_raw.size).convert("RGB").save(result_image_path, format='JPEG', quality=95)
                    LOG(7, "STAGE 2 SUCCESS.")
                    return result_image_path
            except Exception as e2: LOG("!", f"Stage 2 failed: {e2}")

        # STAGE 3: Final Fallback (Original)
        LOG(5, "Stage 3 Fallback triggered.")
        if room_raw: room_raw.save(result_image_path, format='JPEG', quality=90)
        return result_image_path

    except Exception as e_fatal:
        LOG("X", f"Fatal: {e_fatal}")
        try:
            if room_raw: room_raw.save(result_image_path, format='JPEG', quality=90)
            else: Image.new('RGB', (100, 100), color=(255, 255, 255)).save(result_image_path)
        except: pass
        return result_image_path
