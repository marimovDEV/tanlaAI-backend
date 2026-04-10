import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
import requests
# Moved rembg imports inside functions to prevent startup crashes


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
        Background removal using ONLY the free local rembg library.
        No API keys or internet required for this step.
        """
        from .models import Product
        import os
        from django.core.files.base import ContentFile
        try:
            # Refresh instance
            product = Product.objects.get(id=product.id)
            if product.ai_status == 'completed':
                return

            print(f"DEBUG: [AI Service] Processing Background for Product {product.id} (u2net)...")
            product.ai_status = 'processing'
            product.save(update_fields=['ai_status'])

            # 1. Ensure we have original image saved
            if not product.original_image:
                print(f"DEBUG: [AI Service] Initializing original_image from main image for Product {product.id}")
                product.image.seek(0)
                original_content = product.image.read()
                name = os.path.basename(product.image.name)
                product.original_image.save(name, ContentFile(original_content), save=False)
                product.save(update_fields=['original_image'])

            # Prepare image
            product.original_image.seek(0)
            input_image_bytes = product.original_image.read()
            img_pil = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
            
            import numpy as np
            import cv2
            import base64
            import json
            import requests
            from .ai_utils import create_binary_mask, replace_background_with_green

            img_np = np.array(img_pil)
            h, w = img_np.shape[:2]

            # --- STEP 1: GPT-4o Detection ---
            polygon_points = None
            try:
                openai_key = getattr(settings, 'OPENAI_API_KEY', None)
                if not openai_key: raise ValueError("OpenAI Key missing")

                base64_image = base64.b64encode(input_image_bytes).decode('utf-8')
                prompt = 'Identify the main door. Trace outer boundary with a polygon. JSON: {"polygon": [[x, y], ...]} scale 0-1000.'
                
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
                payload = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}],
                    "response_format": {"type": "json_object"}
                }
                
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
                response.raise_for_status()
                content = json.loads(response.json()['choices'][0]['message']['content'])
                poly_data = content.get('polygon')
                if poly_data:
                    polygon_points = [[int(pt[0] * w / 1000), int(pt[1] * h / 1000)] for pt in poly_data]
            except Exception as e:
                print(f"WARNING: [AI Service] GPT-4o failed: {e}")
                # Fallback to a central box if GPT fails to detect
                polygon_points = [[int(w*0.1), int(h*0.05)], [int(w*0.9), int(h*0.05)], [int(w*0.9), int(h*0.95)], [int(w*0.1), int(h*0.95)]]

            # --- STEP 2: Gemini (Nano Banana) or Local Fallback ---
            rgba = None
            try:
                # 2.1 Gemini Green Screen
                mask_io = create_binary_mask(w, h, polygon=polygon_points, invert=True)
                green_screen_bytes = replace_background_with_green(input_image_bytes, mask_io.getvalue())
                
                # 2.2 OpenCV Chroma Key
                nparr = np.frombuffer(green_screen_bytes, np.uint8)
                img_green = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                green_mask = cv2.inRange(img_green, np.array([0, 180, 0]), np.array([140, 255, 140]))
                door_mask = cv2.bitwise_not(green_mask)
                door_mask = cv2.morphologyEx(door_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                
                b, g, r = cv2.split(img_green)
                rgba = cv2.merge((b, g, r, door_mask))
                print(f"DEBUG: [AI Service] Triple-AI Success for {product.id}")
            except Exception as gemini_err:
                print(f"WARNING: [AI Service] Gemini/Chroma failed: {gemini_err}")
                
                # FALLBACK 1: Local rembg (Extremely reliable, offline)
                try:
                    print(f"DEBUG: [AI Service] Activating Local rembg Fallback for {product.id}...")
                    import rembg
                    session = rembg.new_session("u2net")
                    output_bytes = rembg.remove(input_image_bytes, session=session)
                    nparr = np.frombuffer(output_bytes, np.uint8)
                    rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    print(f"DEBUG: [AI Service] Local rembg SUCCESS for {product.id}")
                except Exception as rembg_err:
                    print(f"ERROR: [AI Service] Local rembg ALSO failed: {rembg_err}")
                    
                    # FALLBACK 2: Direct GPT-Polygon Cut (Final Effort)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], (255))
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    b, g, r = cv2.split(img_bgr)
                    rgba = cv2.merge((b, g, r, mask))

            # Encode and Save Result
            if rgba is not None:
                _, final_encoded = cv2.imencode('.png', rgba)
                output_image_bytes = final_encoded.tobytes()
                product.image.save(f"isolated_{product.id}.png", ContentFile(output_image_bytes), save=False)
                product.image_no_bg.save(f"trans_{product.id}.png", ContentFile(output_image_bytes), save=False)
                product.ai_status = 'completed'
                product.save()
                print(f"DEBUG: [AI Service] Final result saved for {product.id}")
            else:
                raise ValueError("All AI pipelines failed to produce an image.")

        except Exception as e:
            print(f"ERROR: [AI Service] UNRECOVERABLE AI Failure for {product.id}: {e}")
            import traceback
            traceback.print_exc()
            product.ai_status = 'error'
            product.save(update_fields=['ai_status'])

        except Exception as e:
            print(f"ERROR: [AI Service] Background removal failed: {e}")
            import traceback
            traceback.print_exc()
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
        from .ai_utils import create_binary_mask, visualize_door_in_room
        
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
