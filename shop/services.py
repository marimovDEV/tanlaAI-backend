import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings


class AIService:
    """Service layer for all AI operations — background removal and room visualization."""
    
    @staticmethod
    def get_gemini_client():
        """Initialize Vertex AI client with service account credentials."""
        import json
        import re
        from google.oauth2 import service_account
        from google import genai

        project = getattr(settings, 'VERTEX_AI_PROJECT', 'ai-image-editor-492616')
        location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')
        key_path = settings.GOOGLE_APPLICATION_CREDENTIALS

        with open(key_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # Robust PEM normalization to handle common paste/formatting issues
        pk = info['private_key']
        # Handle literal "\n" strings that might result from manual file creation
        pk = pk.replace('\\n', '\n')
        
        # Extract the base64 body and re-format it with 64-char lines
        match = re.search(r'-----BEGIN PRIVATE KEY-----(.*)-----END PRIVATE KEY-----', pk, re.DOTALL)
        if match:
            # Strip all whitespace and literal newline characters from the body
            body = "".join(re.findall(r'[A-Za-z0-9+/=]', match.group(1)))
            formatted_body = "\n".join(body[i:i+64] for i in range(0, len(body), 64))
            info['private_key'] = f"-----BEGIN PRIVATE KEY-----\n{formatted_body}\n-----END PRIVATE KEY-----\n"
        else:
            # If no tags found, assume the whole thing needs clean up (less likely but safer)
            info['private_key'] = pk

        credentials = service_account.Credentials.from_service_account_info(
            info, 
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials
        )

    @staticmethod
    def process_product_background(product):
        """
        Auto background removal for new products in target categories.
        Called by signals after product creation.
        """
        from google.genai import types

        try:
            from .models import Product
            # Refresh instance from DB to ensure we have the latest state
            product = Product.objects.get(id=product.id)

            if product.ai_status == 'completed' or product.ai_status == 'error':
                return

            print(f"DEBUG: [AI Service] Processing background removal for Product {product.id}...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # Preserve original
            if not product.original_image:
                image_name = os.path.basename(product.image.name)
                product.image.seek(0)
                original_content = product.image.read()
                product.original_image.save(image_name, ContentFile(original_content), save=False)
            product.save()

            client = AIService.get_gemini_client()

            with open(product.original_image.path, "rb") as f:
                f.seek(0)
                image_bytes = f.read()

            print(f"DEBUG: [AI Service] Calling Gemini Imagen API for Product {product.id}...")
            
            # Use HttpOptions for timeout (120 seconds)
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
                    edit_mode='EDIT_MODE_INPAINT_REMOVAL',
                    number_of_images=1,
                    output_mime_type='image/png',
                    http_options=types.HttpOptions(timeout=120000) # 120 seconds in ms
                )
            )

            if not response.generated_images:
                raise ValueError("Gemini failed to generate isolated image.")

            print(f"DEBUG: [AI Service] API Response received. Saving images for Product {product.id}...")
            generated_img = response.generated_images[0]
            img_data = io.BytesIO(generated_img.image.image_bytes)

            product.image.save(f"processed_{product.id}.png", ContentFile(img_data.getvalue()), save=False)
            product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(img_data.getvalue()), save=False)

            product.ai_status = 'completed'
            product.save(update_fields=['image', 'image_no_bg', 'ai_status'])
            print(f"DEBUG: [AI Service] Background removal success for Product {product.id}.")

        except Exception as e:
            print(f"DEBUG: [AI Service] Background removal error: {e}")
            product.ai_status = 'error'
            product.save(update_fields=['ai_status'])

    @staticmethod
    def generate_room_preview(product, room_image_path, result_image_path):
        """
        Generate door-in-room visualization using Gemini/Imagen.
        Includes retry logic for reliability.
        """
        from google.genai import types

        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                client = AIService.get_gemini_client()

                door_source = product.image_no_bg if product.image_no_bg else product.image

                with open(room_image_path, "rb") as rf:
                    room_bytes = rf.read()
                with open(door_source.path, "rb") as df:
                    door_bytes = df.read()

                print(f"DEBUG: [AI Service] Room visualization attempt {attempt + 1} for Product {product.id}...")

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
                        edit_mode='EDIT_MODE_INPAINT_INSERTION',
                        number_of_images=1,
                        http_options=types.HttpOptions(timeout=120000)
                    )
                )

                if not response.generated_images:
                    raise ValueError("Gemini failed to generate visualization.")

                generated_img = response.generated_images[0]
                final_img = Image.open(io.BytesIO(generated_img.image.image_bytes))
                final_img.save(result_image_path, format='PNG')

                print(f"DEBUG: [AI Service] Visualization success: {result_image_path}")
                return  # Success — exit

            except Exception as e:
                last_error = e
                print(f"DEBUG: [AI Service] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Brief pause before retry

        # All retries failed
        raise last_error


class WishlistService:
    """Service layer for wishlist operations."""

    @staticmethod
    def toggle(user, product):
        """Toggle wishlist status. Returns (is_wishlisted: bool)."""
        from shop.models import Wishlist
        item, created = Wishlist.objects.get_or_create(user=user, product=product)
        if not created:
            item.delete()
            return False
        return True

    @staticmethod
    def is_wishlisted(user_id, product_id):
        """Check if a product is in user's wishlist."""
        from shop.models import Wishlist
        return Wishlist.objects.filter(user_id=user_id, product_id=product_id).exists()

    @staticmethod
    def get_user_wishlist(user):
        """Get all wishlist items for a user."""
        from shop.models import Wishlist
        return Wishlist.objects.filter(user=user).select_related(
            'product', 'product__category', 'product__company'
        )
