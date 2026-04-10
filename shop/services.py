import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.conf import settings
import requests
# Moved rembg imports inside functions to prevent startup crashes


def build_mask_from_polygon(width, height, polygon_points):
    import cv2
    import numpy as np

    if not polygon_points or len(polygon_points) < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def refine_product_mask(mask):
    """
    Convert a noisy alpha/mask into one solid silhouette so door inserts,
    glass cutouts, and decorative white areas are not punched out.
    """
    import cv2
    import numpy as np

    if mask is None:
        return None

    clean = np.where(mask > 10, 255, 0).astype(np.uint8)
    if not np.any(clean):
        return None

    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_labels <= 1:
        return clean

    height, width = clean.shape[:2]
    image_center = (width / 2.0, height / 2.0)
    best_label = None
    best_score = -1.0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area <= 0:
            continue

        component_center = (x + (w / 2.0), y + (h / 2.0))
        distance = ((component_center[0] - image_center[0]) ** 2 + (component_center[1] - image_center[1]) ** 2) ** 0.5
        score = float(area) - (distance * 2.0)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return clean

    clean = np.where(labels == best_label, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return clean

    filled = np.zeros_like(clean)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)
    return filled


def mask_stats(mask):
    import cv2

    if mask is None:
        return 0, 0.0, (0, 0, 0, 0)

    area = int(cv2.countNonZero(mask))
    height, width = mask.shape[:2]
    area_ratio = area / float(max(1, height * width))
    if area == 0:
        return area, area_ratio, (0, 0, 0, 0)
    bbox = cv2.boundingRect(mask)
    return area, area_ratio, bbox


def is_reasonable_door_mask(mask):
    if mask is None:
        return False

    area, area_ratio, bbox = mask_stats(mask)
    if area == 0:
        return False

    _, _, bbox_width, bbox_height = bbox
    height, width = mask.shape[:2]
    height_ratio = bbox_height / float(max(1, height))
    width_ratio = bbox_width / float(max(1, width))

    return 0.03 <= area_ratio <= 0.95 and height_ratio >= 0.35 and width_ratio >= 0.12


def merge_candidate_masks(primary_mask, polygon_mask):
    import cv2

    primary_mask = refine_product_mask(primary_mask)
    polygon_mask = refine_product_mask(polygon_mask)

    if primary_mask is None:
        return polygon_mask
    if polygon_mask is None:
        return primary_mask
    if not is_reasonable_door_mask(polygon_mask):
        return primary_mask

    primary_area, _, _ = mask_stats(primary_mask)
    polygon_area, _, _ = mask_stats(polygon_mask)
    overlap = cv2.countNonZero(cv2.bitwise_and(primary_mask, polygon_mask))
    smaller_area = max(1, min(primary_area, polygon_area))

    # Reject obviously unrelated polygons while still allowing slightly loose unions.
    if overlap / float(smaller_area) < 0.10 and polygon_area > primary_area * 2.5:
        return primary_mask

    return refine_product_mask(cv2.bitwise_or(primary_mask, polygon_mask))


def compose_rgba_from_mask(rgb_image, alpha_mask):
    import cv2

    refined_mask = refine_product_mask(alpha_mask)
    if refined_mask is None:
        return None

    img_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    return cv2.merge((b, g, r, refined_mask))


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
        Orchestrates the background removal process using the new Photoroom-powered engine.
        Cleaned up all legacy GPT-4o/rembg/OpenCV logic.
        """
        from .ai_utils import process_product_with_ai
        # Explicitly call the Photoroom-based processor
        success = process_product_with_ai(product)
        if success:
            print(f"DEBUG: [AIService] Success! Product {product.id} processed via Photoroom.")
        else:
            print(f"ERROR: [AIService] Photoroom processing failed for {product.id}.")

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
                
                prompt = """
                Detect the precise rectangular door opening (door frame area) in this room where a new door can be installed. 
                Return JSON: {"box_2d": [ymin, xmin, ymax, xmax]} scale 0-1000. 
                Focus strictly on the area between the wall opening. Do not include the whole wall.
                """
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
            # --- New v4 Point-Based Precision ---
            try:
                # 1. Calculate door's native aspect ratio
                door_ar = 0.4 
                door_path = product.original_image.path if product.original_image else product.image.path
                with Image.open(door_path) as dp:
                    dw, dh = dp.size
                    door_ar = dw / float(dh)
                
                # 2. Get Top and Bottom points from Gemini + Validation
                point_prompt = """
                Task:
                1. Check if this image is a photo of a real room interior (bedroom, living room, corridor, etc.).
                2. If it is NOT a room (e.g., a flyer, poster, graphic, outdoor, or object photo), return: {"is_room": false}
                3. If it IS a room, identify the [y, x] coordinates for the TOP-CENTER point and BOTTOM-CENTER point of the rectangular door opening.
                Return JSON in this format: {"is_room": true, "top_center": [y, x], "bottom_center": [y, x]} scale 0-1000.
                """
                
                resp = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[point_prompt, types.Part.from_bytes(data=room_bytes, mime_type='image/jpeg')]
                )
                
                import json, re
                text = resp.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                pts = json.loads(match.group()) if match else {}
                
                if not pts.get("is_room", True):
                    print("DEBUG: [AI Service] Image rejected: Not a room interior.")
                    raise ValueError("Ushbu rasm xona interyeriga o'xshamaydi. Iltimos, xonaning real rasmini yuklang.")

                t_y, t_x = pts.get("top_center", [200, 500])
                b_y, b_x = pts.get("bottom_center", [850, 500])
                
                # --- v9 Industrial Floor-Anchored Validation ---
                raw_h = b_y - t_y
                
                # Height must be realistic (not full floor-to-ceiling) fallback
                if raw_h > 850 or raw_h < 200:
                    raw_h = 700
                    mid_y = (t_y + b_y) / 2
                    b_y = mid_y + 350 # Re-anchor floor
                
                # Rule 1: Anchor to Floor (ymax = b_y)
                # Apply 5% reduction for top-margin realism
                target_h = raw_h * 0.95
                target_w = target_h * door_ar * 1.05 # 5% frame buffer for width to cover opening
                
                avg_x = (t_x + b_x) / 2
                
                # Reconstruct Box: Pin TO FLOOR
                box = [
                    b_y - target_h,           # ymin (calculated UP from floor)
                    max(0, avg_x - target_w/2),# xmin
                    b_y,                       # ymax (STAYS AT FLOOR)
                    min(1000, avg_x + target_w/2) # xmax
                ]
                
                print(f"DEBUG: [AI Service v9] Floor-Anchored Box: {box}")

            except Exception as e:
                print(f"DEBUG: [AI Service v9] Validation failed, using default: {e}")
                box = [150, 420, 850, 580]

            # Execute real AI visualization
            visualize_door_in_room(
                product=product,
                room_image_path=room_image_path,
                result_image_path=result_image_path,
                box_1000=box
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
