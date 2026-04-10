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
    Uses Gemini (Nano Banana) to isolate the door and put it on a white background.
    """
    try:
        if product.ai_status != 'none':
            return
            
        print(f"DEBUG: [Gemini AI] Processing creation for Product {product.id}...")
        product.ai_status = 'processing'
        product.save(update_fields=['ai_status'])
        
        # Ensure original image is saved
        if not product.original_image:
            image_name = os.path.basename(product.image.name)
            original_content = product.image.read()
            product.original_image.save(image_name, ContentFile(original_content), save=False)
        product.save()

        client = get_gemini_client()
        
        # Call Gemini for Background Removal (Nano Banana)
        # Using Imagen 3 capability for professional isolation
        with open(product.original_image.path, "rb") as f:
            image_bytes = f.read()

        print("DEBUG: [Gemini/Imagen] Executing background removal...")
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt="Isolate the door product. Output a clean version of the door centered on a pure white background suitable for an e-commerce catalog.",
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=image_bytes),
                    reference_id=0
                )
            ],
            config=types.EditImageConfig(
                edit_mode='BACKGROUND_REMOVAL',
                number_of_images=1,
                output_mime_type='image/png'
            )
        )

        if not response.generated_images:
            raise ValueError("Gemini failed to generate isolated image.")

        # Save the result
        generated_img = response.generated_images[0]
        img_data = io.BytesIO(generated_img.image.image_bytes)
        
        # Update product image and transparent version
        product.image.save(f"processed_{product.id}.png", ContentFile(img_data.getvalue()), save=False)
        product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(img_data.getvalue()), save=False)
        
        product.ai_status = 'completed'
        product.save(update_fields=['image', 'image_no_bg', 'ai_status'])
        print(f"DEBUG: [Gemini AI] Success for Product {product.id}.")

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
    Hybrid Perfection Visualization (v5):
    1. Manually overlay door onto room using PIL (Precision scaling & placement).
    2. Use Gemini AI to harmonize the composite (Shadows, lighting, and edge blending).
    """
    try:
        from PIL import Image, ImageOps, ImageDraw
        import numpy as np
        import io
        client = get_gemini_client()
        
        # 1. Load Original Room & Door
        room_img = Image.open(room_image_path).convert("RGB")
        rw, rh = room_img.size
        
        door_source = product.image_no_bg if product.image_no_bg else product.image
        door_img = Image.open(door_source.path).convert("RGBA")
        
        # 2. Precise Manual Overlay (Python Control)
        if not box_1000:
            box_1000 = [200, 400, 850, 600]
            
        ymin, xmin, ymax, xmax = box_1000
        left, top = int(xmin * rw / 1000), int(ymin * rh / 1000)
        right, bottom = int(xmax * rw / 1000), int(ymax * rh / 1000)
        
        target_h = bottom - top
        target_w = right - left
        
        # Resize door to fit target height exactly, maintain its aspect ratio
        dw, dh = door_img.size
        door_ratio = dw / float(dh)
        resized_w = int(target_h * door_ratio)
        resized_h = target_h
        
        door_resized = door_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        
        # Calculate centering for the width
        center_x = (left + right) // 2
        final_left = center_x - (resized_w // 2)
        final_top = top
        
        # Create dirty composite (Overlay)
        dirty_composite = room_img.copy()
        dirty_composite.paste(door_resized, (final_left, final_top), door_resized)
        
        # --- v9 Pure-Python High Precision Overlay (Imagen 3 Commented Out) ---
        from PIL import ImageFilter
        
        # 3.1 Smart Alpha Trimming (Get actual wood dimensions)
        actual_bbox = door_resized.getbbox()
        if actual_bbox:
            # We don't necessarily want to crop the resized door IF our coordinates 
            # already account for a full-frame door. But usually, trimming helps accuracy.
            pass

        # 3.2 Create Edge Blending Mask
        # We use a blurred mask to blend the door edges smoothly into the room
        mask = Image.new("L", (resized_w, resized_h), 0)
        draw_mask = ImageDraw.Draw(mask)
        # Inner rectangle for the mask (slightly smaller than door to avoid hard edges)
        border = 2
        draw_mask.rectangle([border, border, resized_w - border, resized_h - border], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        # 3.3 Create Synthetic Drop Shadow
        # We create a dark, blurred version of the door alpha to act as a shadow
        shadow_img = Image.new("RGBA", (resized_w + 40, resized_h + 40), (0, 0, 0, 0))
        shadow_color = (0, 0, 0, 90) # Dark translucent
        shadow_mask = door_resized.split()[3] # Take alpha from door
        shadow_part = Image.new("RGBA", (resized_w, resized_h), shadow_color)
        shadow_part.putalpha(shadow_mask)
        shadow_part = shadow_part.filter(ImageFilter.GaussianBlur(radius=10)) # Soft shadow
        
        # 3.4 Compositing
        full_result = room_img.copy()
        
        # Paste Shadow (slightly offset)
        # Offset shadow slightly to the right and down for depth
        full_result.paste(shadow_part, (final_left + 5, final_top + 5), shadow_part)
        
        # Paste Door with feathering mask
        full_result.paste(door_resized, (final_left, final_top), mask)
        
        # --- IMAGEN 3 HARMONIZATION (COMMENTED OUT FOR MANUAL TEST) ---
        """
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
        
        print(f"DEBUG: [Hybrid v5] Harmonizing door {product.id} at ROI {roi_w}x{roi_h}...")
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt=(
                "Harmonize this wood door into the room naturally. Add realistic shadows and ambient occlusion. "
                "Blend the door edges with the wall opening perfectly. Maintain the room's global lighting."
            ),
            reference_images=[
                types.RawReferenceImage(reference_image=types.Image(image_bytes=roi_buf.getvalue()), reference_id=0)
            ],
            config=config
        )
        ...
        """
        
        full_result.save(result_image_path, format='JPEG', quality=95)
        print(f"DEBUG: [Manual v9] Success! Applied Blending and Drop Shadow. Resolution: {rw}x{rh}")
        return result_image_path

    except Exception as e:
        print(f"DEBUG: [Manual v9] Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

    except Exception as e:
        print(f"DEBUG: [Gemini ROI-v2] Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        raise e
