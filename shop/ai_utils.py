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

def is_alpha_mask_valid(img_rgba):
    """Checks if the alpha channel is reasonable (not empty and not a solid block)."""
    import numpy as np
    alpha = np.array(img_rgba.split()[-1])
    total_pixels = alpha.size
    transparent_pixels = np.count_nonzero(alpha < 10)
    visible_pixels = np.count_nonzero(alpha > 200)
    
    # If more than 98% is transparent, it's likely a failed segmentation
    if transparent_pixels / total_pixels > 0.98:
        return False
    # If everything is solid, it's not a transparent PNG
    if visible_pixels / total_pixels > 0.99:
        return False
    return True

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None):
    """
    Bulletproof Hybrid (v14):
    1. Multi-stage Asset Loading (Validation + Fallback).
    2. Precise Manual Overlay.
    3. AI Harmonization with guaranteed Python fallback.
    """
    try:
        from PIL import Image, ImageOps, ImageDraw
        import numpy as np
        import io
        import rembg
        client = get_gemini_client()
        
        # 1. Load Room
        room_img = Image.open(room_image_path).convert("RGB")
        rw, rh = room_img.size
        
        # 2. Bulletproof Asset Loading
        door_img = None
        candidates = [
            (product.image_no_bg, "isolated PNG"),
            (product.image, "standard image"),
            (product.original_image, "original raw")
        ]
        
        for field, label in candidates:
            if field and field.name and os.path.exists(field.path):
                try:
                    candidate_img = Image.open(field.path).convert("RGBA")
                    if is_alpha_mask_valid(candidate_img):
                        door_img = candidate_img
                        print(f"DEBUG: [Bulletproof] Using {label} as valid asset.")
                        break
                    else:
                        print(f"DEBUG: [Bulletproof] {label} failed alpha validation.")
                except Exception as e:
                    print(f"DEBUG: [Bulletproof] Error loading {label}: {e}")

        # Final extreme fallback: Re-render alpha from original if all else fails
        if not door_img and product.original_image:
            try:
                print("DEBUG: [Bulletproof] All assets invalid. Attempting emergency rembg on original...")
                with open(product.original_image.path, "rb") as f:
                    orig_bytes = f.read()
                no_bg_bytes = rembg.remove(orig_bytes)
                door_img = Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
            except Exception as e:
                print(f"DEBUG: [Bulletproof] Emergency rembg failed: {e}")

        if not door_img:
            raise ValueError("No valid door asset found for visualization.")

        # 3. Precise Manual Overlay
        if not box_1000:
            box_1000 = [200, 400, 850, 600]
            
        ymin, xmin, ymax, xmax = box_1000
        left, top = int(xmin * rw / 1000), int(ymin * rh / 1000)
        right, bottom = int(xmax * rw / 1000), int(ymax * rh / 1000)
        
        target_h = bottom - top
        target_w = right - left
        
        # Deep Anchoring (b_y + 15) and Width Calibration (min 0.43 ratio)
        dw, dh = door_img.size
        door_ar = dw / float(dh)
        
        # Recalculate target width based on product aspect ratio for realism
        resized_h = target_h
        resized_w = int(resized_h * door_ar)
        
        # Safety: don't let it be too thin
        if resized_w < (resized_h * 0.43):
            resized_w = int(resized_h * 0.43)

        door_resized = door_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        
        center_x = (left + right) // 2
        final_left = center_x - (resized_w // 2)
        # Deep Anchor: lower the door slightly into the floor to avoid floating
        final_top = top + 15 
        
        dirty_composite = room_img.copy()
        dirty_composite.paste(door_resized, (final_left, final_top), door_resized)
        
        # 4. ROI for AI Harmonization
        side = int(max(resized_w, resized_h) * 1.5)
        side = min(side, rw, rh)
        
        roi_left = max(0, center_x - side // 2)
        roi_top = max(0, final_top + resized_h//2 - side // 2)
        
        if roi_left + side > rw: roi_left = rw - side
        if roi_top + side > rh: roi_top = rh - side
        roi_left, roi_top = max(0, roi_left), max(0, roi_top)
        
        roi_box = (roi_left, roi_top, roi_left + side, roi_top + side)
        roi_img = dirty_composite.crop(roi_box)
        roi_w, roi_h = roi_img.size
        
        local_mask = Image.new("L", (roi_w, roi_h), 0)
        draw = ImageDraw.Draw(local_mask)
        m_pad = int(resized_w * 0.35) # Generous padding for shadow inpainting
        m_left = (final_left - roi_left) - m_pad
        m_top = (final_top - roi_top) - m_pad
        m_right = (final_left + resized_w - roi_left) + m_pad
        m_bottom = (final_top + resized_h - roi_top) + m_pad
        draw.rectangle([m_left, m_top, m_right, m_bottom], fill=255)
        
        # 5. AI Call with Guaranteed Fallback
        try:
            roi_buf = io.BytesIO()
            roi_img.save(roi_buf, format='JPEG')
            mask_buf = io.BytesIO()
            local_mask.save(mask_buf, format='PNG')
            
            config = types.EditImageConfig(
                edit_mode='INPAINT_INSERT',
                number_of_images=1,
                output_mime_type='image/png',
                mask=types.Image(image_bytes=mask_buf.getvalue())
            )
            
            print(f"DEBUG: [Super Aggressive v15] Forcing replacement for Product {product.id}...")
            response = client.models.edit_image(
                model='imagen-3.0-capability-001',
                prompt=(
                    "This image already contains an existing door. You MUST: "
                    "1. Find the existing door in the masked area. "
                    "2. REMOVE it completely from the wall. "
                    "3. REPLACE it with the new provided reference door. "
                    "Do NOT return the original image. Do NOT keep the old door. "
                    "The result must clearly show a DIFFERENT door from the original. "
                    "The new door must be perfectly integrated into the wall, matching the floor perspective."
                ),
                reference_images=[
                    types.RawReferenceImage(reference_image=types.Image(image_bytes=roi_buf.getvalue()), reference_id=0)
                ],
                config=config
            )
            
            if response.generated_images:
                generated_roi_img = Image.open(io.BytesIO(response.generated_images[0].image.image_bytes)).resize((roi_w, roi_h))
                full_result = room_img.copy()
                full_result.paste(generated_roi_img, (roi_left, roi_top))
                full_result.save(result_image_path, format='JPEG', quality=95)
                print(f"DEBUG: [Bulletproof v14] Success with AI Harmonization.")
                return result_image_path
        except Exception as ai_e:
            print(f"WARNING: [Bulletproof v14] AI Harmonization failed: {ai_e}. Falling back to dirty composite.")

        # FINAL FALLBACK (If AI fails or returns nothing)
        dirty_composite.save(result_image_path, format='JPEG', quality=95)
        print(f"DEBUG: [Bulletproof v14] Used Dirty Composite as fallback.")
        return result_image_path

    except Exception as e:
        print(f"ERROR: [Bulletproof v14] Fatal crash: {e}")
        import traceback
        traceback.print_exc()
        raise e

    except Exception as e:
        print(f"DEBUG: [Gemini ROI-v2] Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        raise e
