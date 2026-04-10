import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings


class AIService:
    """Service layer for all AI operations — background removal and room visualization."""
    
    @staticmethod
    def get_gemini_client():
        """Initialize Gemini client using API Key (primary) or Service Account (fallback)."""
        import json
        from google import genai
        from google.oauth2 import service_account

        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        
        if api_key:
            print("DEBUG: [AI Service] Initializing client with API KEY...")
            return genai.Client(api_key=api_key)
        
        # Fallback to Service Account for Vertex AI features
        key_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        if os.path.exists(key_path):
            print("DEBUG: [AI Service] Initializing client with SERVICE ACCOUNT (Vertex AI)...")
            project = getattr(settings, 'VERTEX_AI_PROJECT', '')
            location = getattr(settings, 'VERTEX_AI_LOCATION', 'us-central1')
            
            with open(key_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            
            pk = info.get('private_key', '')
            info['private_key'] = pk.replace('\\n', '\n')

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
        
        raise ValueError("Neither GEMINI_API_KEY nor valid google-cloud-key.json found.")

    @staticmethod
    def process_product_background(product):
        """
        Robust background removal using rembg (local) and Gemini (smart check).
        """
        import rembg
        from .models import Product
        
        try:
            # Refresh instance
            product = Product.objects.get(id=product.id)
            if product.ai_status == 'completed':
                return

            print(f"DEBUG: [AI Service] Processing background removal for Product {product.id} via rembg...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # Ensure we have original image
            if not product.original_image:
                product.image.seek(0)
                original_content = product.image.read()
                product.original_image.save(os.path.basename(product.image.name), ContentFile(original_content), save=False)
            
            product.original_image.seek(0)
            input_image_bytes = product.original_image.read()
            
            # Local background removal with high quality settings
            print("DEBUG: [AI Service] Performing high-quality local background removal (alpha matting)...")
            output_image_bytes = rembg.remove(
                input_image_bytes,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10
            )
            
            # Save the isolated product image
            image_name = f"isolated_{product.id}.png"
            product.image.save(image_name, ContentFile(output_image_bytes), save=False)
            product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(output_image_bytes), save=False)
            
            product.ai_status = 'completed'
            product.save()
            print(f"DEBUG: [AI Service] Background removal COMPLETED locally for Product {product.id}")

        except Exception as e:
            print(f"ERROR: [AI Service] Background removal failed: {e}")
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
