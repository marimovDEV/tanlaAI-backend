import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
import rembg
import requests
from rembg import new_session
from .ai_utils import visualize_door_in_room, create_binary_mask


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
        Background removal using remove.bg API (preferred) with rembg (fallback).
        """
        from .models import Product
        try:
            # Refresh instance
            product = Product.objects.get(id=product.id)
            if product.ai_status == 'completed':
                return

            print(f"DEBUG: [AI Service] Processing Background for Product {product.id}...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # Ensure we have original image
            if not product.original_image:
                product.image.seek(0)
                original_content = product.image.read()
                product.original_image.save(os.path.basename(product.image.name), ContentFile(original_content), save=False)

            api_key = getattr(settings, 'REMOVE_BG_API_KEY', '')
            output_image_bytes = None

            # 1. Try remove.bg API if key is available
            if api_key and api_key != 'YOUR_API_KEY_HERE':
                try:
                    print("DEBUG: [AI Service] Attempting background removal via remove.bg API...")
                    product.original_image.seek(0)
                    response = requests.post(
                        'https://api.remove.bg/v1.0/removebg',
                        files={'image_file': product.original_image.open('rb')},
                        data={'size': 'auto'},
                        headers={'X-Api-Key': api_key},
                        timeout=30
                    )
                    if response.status_code == requests.codes.ok:
                        output_image_bytes = response.content
                        print("DEBUG: [AI Service] remove.bg API SUCCESS.")
                    else:
                        print(f"DEBUG: [AI Service] remove.bg API failed (status {response.status_code}): {response.text}")
                except Exception as api_err:
                    print(f"DEBUG: [AI Service] remove.bg API connection error: {api_err}")

            # 2. Fallback to local rembg if API failed or was skipped
            if not output_image_bytes:
                print("DEBUG: [AI Service] Falling back to local rembg (isnet-general-use)...")
                session = new_session("isnet-general-use")
                product.original_image.seek(0)
                input_image_bytes = product.original_image.read()
                output_image_bytes = rembg.remove(
                    input_image_bytes,
                    session=session,
                    alpha_matting=False,
                    post_process_mask=True
                )

            # Save results
            image_name = f"isolated_{product.id}.png"
            product.image.save(image_name, ContentFile(output_image_bytes), save=False)
            product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(output_image_bytes), save=False)
            
            product.ai_status = 'completed'
            product.save()
            print(f"DEBUG: [AI Service] Background removal COMPLETED for Product {product.id}")

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
        import json
        import re
        
        # Flag to indicate if we have custom boxes
        box = [200, 300, 850, 700]  # Default central box [ymin, xmin, ymax, xmax]
        
        # 1. Try to get AI coordinates, but don't let failures stop us
        try:
            client = AIService.get_gemini_client()
            if client:
                with open(room_image_path, "rb") as f:
                    room_bytes = f.read()
                
                prompt = "Return JSON: {\"box_2d\": [ymin, xmin, ymax, xmax]} for the main door frame. Values 0-1000."
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[prompt, types.Part.from_bytes(data=room_bytes, mime_type='image/jpeg')]
                )
                
                if response and response.text:
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if match:
                        ai_box = json.loads(match.group(0)).get('box_2d')
                        if ai_box and len(ai_box) == 4:
                            box = ai_box
                            print(f"DEBUG: [AI Service] Using AI coordinates: {box}")
        except Exception as ai_err:
            print(f"DEBUG: [AI Service] AI detection ignored (falling back to center): {ai_err}")

        # 2. Variant B: AI Inpainting (imagen-3.0)
        try:
            from PIL import Image
            
            # Detect room image size for mask creation
            with Image.open(room_image_path) as room_img:
                width, height = room_img.size
            
            # Create binary mask from detected box
            print(f"DEBUG: [AI Service] Creating mask for box {box} on {width}x{height} image...")
            mask_io = create_binary_mask(width, height, box)
            mask_bytes = mask_io.getvalue()

            # Execute real AI visualization
            visualize_door_in_room(
                product=product,
                room_image_path=room_image_path,
                result_image_path=result_image_path,
                mask_bytes=mask_bytes
            )
            
            print(f"DEBUG: [AI Service] AI Inpainting SUCCESS for {product.name}")
            return result_image_path

        except Exception as e:
            print(f"ERROR: [AI Service] AI Inpainting failed: {e}")
            
            # 3. Final Fallback: Simple PIL Overlay if AI inpainting fails
            try:
                print("DEBUG: [AI Service] Falling back to Variant A (PIL Overlay)...")
                room_img = Image.open(room_image_path).convert("RGBA")
                rw, rh = room_img.size
                
                door_path = product.image.path
                if product.image_no_bg and product.image_no_bg.name:
                    if os.path.exists(product.image_no_bg.path):
                        door_path = product.image_no_bg.path
                
                door_img = Image.open(door_path).convert("RGBA")
                ymin, xmin, ymax, xmax = box
                left, top = int(xmin * rw / 1000), int(ymin * rh / 1000)
                right, bottom = int(xmax * rw / 1000), int(ymax * rh / 1000)
                
                door_resized = door_img.resize((max(1, right-left), max(1, bottom-top)), Image.Resampling.LANCZOS)
                room_img.paste(door_resized, (left, top), door_resized)
                
                final_res = room_img.convert("RGB")
                final_res.save(result_image_path, "JPEG", quality=95)
                return result_image_path
            except Exception as fallback_err:
                print(f"ERROR: [AI Service] Final fallback also failed: {fallback_err}")
                return None


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
