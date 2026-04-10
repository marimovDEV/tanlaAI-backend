import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
import rembg
from rembg import new_session


class AIService:
    """Service layer for all AI operations — background removal and room visualization."""
    
    @staticmethod
    def get_gemini_client():
        """Initialize Gemini client using API Key (primary) or Service Account (fallback)."""
        import json
        from google import genai
        from google.oauth2 import service_account

        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if isinstance(api_key, str):
            api_key = api_key.strip()
        
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
        Robust background removal using rembg (isnet-general-use).
        We use a more conservative approach to preserve door frames/platbands.
        """
        from .models import Product
        try:
            # Refresh instance
            product = Product.objects.get(id=product.id)
            if product.ai_status == 'completed':
                return

            print(f"DEBUG: [AI Service] Processing Background for Product {product.id} (model: isnet-general-use)...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # Ensure we have original image
            if not product.original_image:
                product.image.seek(0)
                original_content = product.image.read()
                product.original_image.save(os.path.basename(product.image.name), ContentFile(original_content), save=False)

            # Load model session
            session = new_session("isnet-general-use")
            
            product.original_image.seek(0)
            input_image_bytes = product.original_image.read()
            
            # Use conservative settings: turn off alpha_matting as it tends to eat wood edges
            output_image_bytes = rembg.remove(
                input_image_bytes,
                session=session,
                alpha_matting=False, # Disable to preserve hard wooden edges
                only_mask=False,
                post_process_mask=True # Clean up the edges naturally
            )
            
            # Save results
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
        Uses Gemini 1.5 Flash to detect door frame coordinates and overlays 
        the product image locally. This bypasses Imagen 3 limitations.
        """
        from google.genai import types
        
        try:
            client = AIService.get_gemini_client()
            
            # 1. Detect door frame coordinates using Gemini 1.5 Flash
            with open(room_image_path, "rb") as f:
                room_bytes = f.read()

            prompt = (
                "You are a computer vision expert. Identify the bounding box of the main doorway, "
                "entrance, or door frame in this room image where a new door should be installed. "
                "Return only a STRICT JSON dictionary: {\"box_2d\": [ymin, xmin, ymax, xmax]}. "
                "Values must be normalized to 1000. If multiple doors exist, pick the most central one. "
                "Output ONLY the JSON, no other text."
            )
            
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=room_bytes, mime_type='image/jpeg')
                    ]
                )
                text = response.text.strip()
                print(f"DEBUG: [AI Service] Gemini detection response: {text}")
                
                import json
                import re
                
                # Extract JSON using simple regex
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                
                if json_match:
                    try:
                        coords = json.loads(json_match.group(0))
                        box = coords.get('box_2d')
                    except json.JSONDecodeError:
                        # Try to find a list like [1, 2, 3, 4] if JSON wrapper is missing/broken
                        list_match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
                        if list_match:
                            box = [int(x) for x in list_match.groups()]
                        else:
                            box = None
                else:
                    box = None
            except Exception as e:
                print(f"WARNING: Gemini detection failed/errored: {e}. Using fallback center box.")
                box = None

            # Fallback: if Gemini fails or returns nothing, use a standard center door box
            if not box or len(box) != 4:
                print("DEBUG: [AI Service] Using fallback center box (150, 350, 850, 650)")
                box = [150, 350, 850, 650]

            ymin, xmin, ymax, xmax = box

            # 2. Local Overlay using PIL
            from PIL import Image
            
            room_img = Image.open(room_image_path).convert("RGBA")
            # Prefer the no-background version we created earlier
            door_source = None
            if product.image_no_bg and product.image_no_bg.name:
                # Check if the file actually exists on disk
                if os.path.exists(product.image_no_bg.path):
                    door_source = product.image_no_bg
            
            if not door_source:
                door_source = product.image
                
            door_img = Image.open(door_source.path).convert("RGBA")

            w, h = room_img.size
            
            # Convert normalized coordinates to pixels
            left = xmin * w / 1000
            top = ymin * h / 1000
            right = xmax * w / 1000
            bottom = ymax * h / 1000
            
            target_w = int(right - left)
            target_h = int(bottom - top)

            if target_w <= 0 or target_h <= 0:
                raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")

            # Resize door to fit the detected frame
            door_resized = door_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            # Paste door onto room
            room_img.alpha_composite(door_resized, (int(left), int(top)))
            
            # Save final result
            room_img.convert("RGB").save(result_image_path, "JPEG", quality=95)
            
            print(f"DEBUG: [AI Service] Visualization success: {result_image_path}")
            return result_image_path

        except Exception as e:
            print(f"ERROR: [AI Service] Visualization failed: {str(e)}")
            raise e


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
