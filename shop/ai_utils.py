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
        # Map coordinates with a VERY GENEROUS 12% buffer to ensure zero clipping
        ymin, xmin, ymax, xmax = box_1000
        
        # Expand box by 12% (120 units in 1000-scale) for maximum safety
        xmin = max(0, xmin - 120)
        ymin = max(0, ymin - 120)
        xmax = min(1000, xmax + 120)
        ymax = min(1000, ymax + 120)

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

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None):
    """
    v19 — Full logging + bugfixes.
    Har bir qadam serverda log qilinadi.
    """
    import time
    start_t = time.time()
    
    def LOG(step, msg):
        elapsed = time.time() - start_t
        print(f"[v19 {elapsed:6.2f}s] STEP {step}: {msg}")
    
    LOG(0, f"=== VIZUALIZATSIYA BOSHLANDI ===")
    LOG(0, f"Product: #{product.id} '{product.name}'")
    LOG(0, f"Room image: {room_image_path}")
    LOG(0, f"Result path: {result_image_path}")
    LOG(0, f"box_1000: {box_1000}")
    
    try:
        from PIL import Image, ImageOps, ImageDraw
        from google.genai import types
        import numpy as np
        import io
        
        # ====== STEP 1: AI Client ======
        LOG(1, "AI Client yaratilmoqda...")
        from shop.services import AIService
        client = AIService.get_gemini_client()
        LOG(1, f"Client tayyor: {type(client).__name__}")
        
        # ====== STEP 2: Room yuklash ======
        LOG(2, "Xona rasmi yuklanmoqda...")
        room_img_raw = Image.open(room_image_path)
        try:
            from PIL import ImageOps
            room_img = ImageOps.exif_transpose(room_img_raw).convert("RGB")
        except Exception:
            room_img = room_img_raw.convert("RGB")
        rw, rh = room_img.size
        LOG(2, f"Xona o'lchami: {rw}x{rh}")
        
        # ====== STEP 3: Eshik asset yuklash ======
        LOG(3, "Eshik asseti yuklanmoqda...")
        door_img = None
        asset_source = None
        
        for attr_name in ['image_no_bg', 'image', 'original_image']:
            field = getattr(product, attr_name, None)
            if not field or not field.name:
                LOG(3, f"  {attr_name}: bo'sh (o'tkazib yuboryapman)")
                continue
            
            try:
                path = field.path
            except Exception:
                LOG(3, f"  {attr_name}: path olishda xato")
                continue
                
            if not os.path.exists(path):
                LOG(3, f"  {attr_name}: fayl TOPILMADI: {path}")
                continue
            
            file_size = os.path.getsize(path) / 1024
            LOG(3, f"  {attr_name}: fayl topildi ({file_size:.0f}KB) — {path}")
            
            try:
                door_img = Image.open(path)
                LOG(3, f"  {attr_name}: ochildi — o'lcham: {door_img.size}, mode: {door_img.mode}")
                door_img = door_img.convert("RGBA")
                LOG(3, f"  {attr_name}: RGBA ga o'tkazildi")
                asset_source = attr_name
                break
            except Exception as e:
                LOG(3, f"  {attr_name}: ochishda XATO: {e}")
                continue
        
        if asset_source != 'image_no_bg' and door_img:
            LOG(3, f"'{asset_source}' fon bilan keldi! rembg orqali fonni avtomatik o'chirmoqdamiz...")
            try:
                from rembg import remove
                b = io.BytesIO()
                door_img.save(b, format='PNG')
                out_b = remove(b.getvalue())
                door_img = Image.open(io.BytesIO(out_b)).convert("RGBA")
                asset_source = 'rembg'
                LOG(3, "rembg fonni MUVAFFAQIYATLI o'chirdi!")
            except Exception as e:
                LOG(3, f"rembg XATOLIGI (o'tkazib yuborilmoqda): {e}")

        if not door_img:
            LOG(3, "XATO: HECH QANDAY ASSET TOPILMADI!")
            raise ValueError("NO IMAGE ASSET FOUND for product #" + str(product.id))
        
        dw, dh = door_img.size
        LOG(3, f"Eshik asseti tayyor: {dw}x{dh} (manba: {asset_source})")
        
        # ====== STEP 4: Joylashtirish matematikasi ======
        LOG(4, "Placement hisob-kitob boshlandi...")
        if not box_1000:
            LOG(4, "box_1000 mavjud emas. GPT-4o orqali eshik o'rni izlanmoqda...")
            try:
                r_bytes = io.BytesIO()
                room_img.save(r_bytes, format='JPEG', quality=85)
                import base64
                import requests
                from django.conf import settings

                base64_image = base64.b64encode(r_bytes.getvalue()).decode('utf-8')
                openai_key = getattr(settings, 'OPENAI_API_KEY', os.getenv("OPENAI_API_KEY"))
                if not openai_key: raise ValueError("OpenAI Key missing. Please set it in .env")

                prompt = (
                    "You are a computer vision system.\n"
                    "Analyze this room image and find the best place where a door can be installed.\n"
                    "Rules:\n"
                    "- Door must be placed ONLY on a wall\n"
                    "- Door must NOT be placed on windows\n"
                    "- Door must be vertical rectangle\n"
                    "- Door must touch the floor\n"
                    "- Door must not overlap furniture\n"
                    "Return ONLY JSON:\n"
                    "{\n"
                    "  \"x\": number (0-1),\n"
                    "  \"y\": number (0-1),\n"
                    "  \"width\": number (0-1),\n"
                    "  \"height\": number (0-1)\n"
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
                
                LOG(4, "Sending room image to GPT...")
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
                response.raise_for_status()
                
                import json
                content = json.loads(response.json()['choices'][0]['message']['content'])
                LOG(4, f"GPT BOX: {content}")
                
                # Validation rule from user
                if content.get("width", 0) < 0.1 or content.get("height", 0) < 0.2 or (content.get("y", 0) + content.get("height", 0)) > 1.0:
                    raise ValueError("GPT box failed validation rules")
                
                box_x = content["x"]
                box_y = content["y"]
                box_w = content["width"]
                box_h = content["height"]
                box_1000 = [
                    int(box_y * 1000), 
                    int(box_x * 1000), 
                    int((box_y + box_h) * 1000), 
                    int((box_x + box_w) * 1000)
                ]
            except Exception as e:
                LOG(4, f"GPT detection XATOLIGI: {e}")
                box_1000 = [200, 400, 850, 600]
                LOG(4, f"Fallback Default box ishlatilmoqda: {box_1000}")
            
        ymin, xmin, ymax, xmax = box_1000
        left = int(xmin * rw / 1000)
        top = int(ymin * rh / 1000)
        right = int(xmax * rw / 1000)
        bottom = int(ymax * rh / 1000)
        LOG(4, f"Original Pixel box: left={left}, top={top}, right={right}, bottom={bottom}")
        
        # CHEAP WINDOW DETECTION HACK
        import numpy as np
        roi = room_img.crop((left, top, right, bottom))
        brightness = np.mean(np.array(roi))
        LOG(4, f"ROI Brightness check: {brightness:.2f}")
        
        if brightness > 190:
            LOG(4, "WARNING: Oyna (Window) aniqlandi (yoki juda yorug' devor). Fallback ishlatilmoqda...")
            left, top, right, bottom = (int(0.40 * rw), int(0.20 * rh), int(0.60 * rw), int(0.85 * rh))
            LOG(4, f"Fallback Pixel box: left={left}, top={top}, right={right}, bottom={bottom}")

        target_h = bottom - top
        door_ar = dw / float(dh)
        
        resized_h = target_h
        resized_w = int(resized_h * door_ar)
        
        if resized_w < (target_h * 0.42):
            resized_w = int(target_h * 0.42)
        if resized_w > (target_h * 0.80):
            resized_w = int(target_h * 0.80)

        door_resized = door_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        LOG(4, f"Eshik resize: {resized_w}x{resized_h}")
        
        center_x = (left + right) // 2
        final_left = center_x - (resized_w // 2)
        
        # Floor snap constraint - using the box bottom instead of image bottom
        final_top = bottom - resized_h
        LOG(4, f"Joylashtirish: final_left={final_left}, final_top={final_top}")
        
        # ====== STEP 5: Dirty Composite ======
        LOG(5, "Dirty composite yaratilmoqda...")
        dirty_composite = room_img.copy()
        if door_resized.mode == 'RGBA':
            dirty_composite.paste(door_resized, (final_left, final_top), door_resized)
            LOG(5, "Alpha-matte paste amalga oshdi")
        else:
            dirty_composite.paste(door_resized, (final_left, final_top))
            LOG(5, "Oddiy paste amalga oshdi (alpha yo'q)")
        
        # Debug: save dirty composite
        debug_dir = os.path.join(os.path.dirname(result_image_path))
        debug_dirty = os.path.join(debug_dir, f'debug_dirty_{product.id}.jpg')
        try:
            dirty_composite.save(debug_dirty, format='JPEG', quality=85)
            LOG(5, f"Debug dirty composite saqlandi: {debug_dirty}")
        except:
            LOG(5, "Debug dirty composite saqlab bo'lmadi")
        
        # ====== STEP 6: ROI va Mask tayyorlash ======
        LOG(6, "ROI va mask tayyorlanmoqda...")
        side = int(max(resized_w, resized_h) * 1.4)
        side = min(side, rw, rh)
        roi_left = max(0, center_x - side // 2)
        roi_top = max(0, final_top + resized_h // 2 - side // 2)
        if roi_left + side > rw: roi_left = rw - side
        if roi_top + side > rh: roi_top = rh - side
        roi_left, roi_top = max(0, roi_left), max(0, roi_top)
        
        roi_box = (roi_left, roi_top, roi_left + side, roi_top + side)
        LOG(6, f"ROI box: {roi_box} (side={side})")
        
        roi_img = dirty_composite.crop(roi_box)
        roi_w, roi_h = roi_img.size
        LOG(6, f"ROI o'lchami: {roi_w}x{roi_h}")
        
        # Mask — faqat chetlarni qamrab oladi
        local_mask = Image.new("L", (roi_w, roi_h), 0)
        draw = ImageDraw.Draw(local_mask)
        m_pad = int(resized_w * 0.10) # 10% outer padding
        m_left = (final_left - roi_left) - m_pad
        m_top_mask = (final_top - roi_top) - m_pad
        m_right = (final_left + resized_w - roi_left) + m_pad
        m_bottom = (final_top + resized_h - roi_top) + int(m_pad * 2) # extend bottom for floor shadows
        
        # Outer mask boundary (white)
        draw.rectangle([m_left, m_top_mask, m_right, m_bottom], fill=255)
        
        # Inner mask boundary (black to protect the door itself!)
        i_pad = int(resized_w * 0.04) # 4% internal protection
        i_left = (final_left - roi_left) + i_pad
        i_top = (final_top - roi_top) + i_pad
        i_right = (final_left + resized_w - roi_left) - i_pad
        i_bottom = (final_top + resized_h - roi_top) - i_pad
        draw.rectangle([i_left, i_top, i_right, i_bottom], fill=0)
        
        LOG(6, f"Mask: ring around door generated to protect original door")
        
        # Serialize
        m_buf = io.BytesIO()
        local_mask.save(m_buf, format='PNG')
        r_buf = io.BytesIO()
        roi_img.save(r_buf, format='PNG')
        LOG(6, f"Bufferlar tayyor: roi={len(r_buf.getvalue())} bytes, mask={len(m_buf.getvalue())} bytes")
        
        # ====== STEP 7: AI Harmonization ======
        LOG(7, "Imagen 3 harmonization SO'ROVI yuborilmoqda...")
        try:
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=(
                    "You are an image editing model.\n"
                    "A door is already placed correctly in the scene.\n"
                    "STRICT RULES:\n"
                    "- DO NOT remove the door\n"
                    "- DO NOT change door shape\n"
                    "- DO NOT resize door\n"
                    "- DO NOT replace door\n"
                    "Your task:\n"
                    "- match lighting\n"
                    "- add realistic shadows\n"
                    "- blend edges smoothly\n"
                    "The door is a rectangular vertical object.\n"
                    "Keep it intact."
                ),
                reference_images=[
                    types.RawReferenceImage(
                        reference_image=types.Image(image_bytes=r_buf.getvalue()),
                        reference_id=1,
                    ),
                    types.MaskReferenceImage(
                        reference_id=2,
                        reference_image=types.Image(image_bytes=m_buf.getvalue()),
                        config=types.MaskReferenceConfig(
                            mask_mode='MASK_MODE_USER_PROVIDED',
                            mask_dilation=0.03,
                        ),
                    ),
                ],
                config=types.EditImageConfig(
                    edit_mode='EDIT_MODE_INPAINT_INSERTION',
                    number_of_images=1,
                    output_mime_type='image/png',
                ),
            )
            LOG(7, f"Imagen 3 javob berdi!")
            
            if response.generated_images:
                img_bytes = response.generated_images[0].image.image_bytes
                LOG(7, f"Natija rasmi bor: {len(img_bytes)} bytes")
                
                gen_roi = Image.open(io.BytesIO(img_bytes)).resize((roi_w, roi_h))
                full = room_img.copy()
                full.paste(gen_roi, (roi_left, roi_top))
                full.save(result_image_path, format='JPEG', quality=95)
                LOG(7, f"MUVAFFAQIYATLI! Natija saqlandi: {result_image_path}")
                return result_image_path
            else:
                LOG(7, "Imagen 3 rasm QAYTARMADI (empty response)")
                
        except Exception as e:
            LOG(7, f"AI harmonization XATO: {e}")
            import traceback
            traceback.print_exc()

        # ====== STEP 8: Fallback ======
        LOG(8, "Fallback: dirty composite saqlanmoqda...")
        dirty_composite.save(result_image_path, format='JPEG', quality=90)
        LOG(8, f"Fallback natija saqlandi: {result_image_path}")
        return result_image_path

    except Exception as e:
        LOG("X", f"💥 FATAL XATO: {e}")
        import traceback
        traceback.print_exc()
        raise e

