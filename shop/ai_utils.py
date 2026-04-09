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

def visualize_door_in_room(product, room_image_path, result_image_path):
    """
    Uses Gemini (Nano Banana) to realistically install a door into a room photo in one pass.
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
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt=(
                "Install the door from image 1 into the most suitable door frame or entrance wall found in image 0. "
                "Ensure correct scale, perspective, and realistic shadows. The final image should look like a real home photo."
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
            config=types.EditImageConfig(
                edit_mode='INPAINT_INSERT',
                number_of_images=1
            )
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
