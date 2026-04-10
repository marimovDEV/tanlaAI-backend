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

def visualize_door_in_room(product, room_image_path, result_image_path, mask_bytes=None):
    """
    Uses Gemini/Imagen to realistically install a door into a room photo using inpainting.
    If mask_bytes is provided, it uses it to guide the insertion.
    """
    try:
        client = get_gemini_client()
        
        # Use a high-quality door image (original or isolated)
        door_source = product.image_no_bg if product.image_no_bg else product.image
        
        with open(room_image_path, "rb") as rf:
            room_bytes = rf.read()
        with open(door_source.path, "rb") as df:
            door_bytes = df.read()

        print(f"DEBUG: [Gemini/Nano Banana] Visualizing door {product.id} in room...")
        
        # Smart inpainting/insertion call
        config = types.EditImageConfig(
            edit_mode='INPAINT_INSERT',
            number_of_images=1,
            output_mime_type='image/png'
        )
        
        if mask_bytes:
            config.mask = types.Image(image_bytes=mask_bytes)

        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt=(
                "Professionally install the high-quality wood door from image 1 into the specified door opening area in image 0. "
                "Match the room's lighting, perspective, and shadows perfectly. The door should be perfectly aligned within the door frame."
            ),
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=room_bytes),
                    reference_id=0
                ),
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=door_bytes),
                    reference_id=1
                )
            ],
            config=config
        )

        if not response.generated_images:
            raise ValueError("Gemini failed to generate visualization.")

        # Save final result
        generated_img = response.generated_images[0]
        final_img = Image.open(io.BytesIO(generated_img.image.image_bytes))
        final_img.save(result_image_path, format='PNG')
        
        print(f"DEBUG: [Gemini AI] Visualization successful: {result_image_path}")

    except Exception as e:
        print(f"DEBUG: [Gemini AI] Error during visualization: {e}")
        # If it fails, we can handle it in the polling view
        raise e
