"""
DALL-E 3 + GPT-4o Door Visualization.
Proven pipeline: GPT-4o analyzes both images, DALL-E 3 generates the final result.
"""
import os
import io
import base64
import tempfile
import traceback
import requests
from PIL import Image
from openai import OpenAI
from django.conf import settings


def log_error(msg):
    import time
    try:
        with open('ai_error.log', 'a') as f:
            f.write(f"[{time.ctime()}] {msg}\n")
    except:
        pass
    print(msg)


def load_visualization_metadata(image_path):
    """Legacy stub - not used in new pipeline."""
    return None


def _get_openai_client():
    """Get OpenAI client with API key from settings."""
    api_key = getattr(settings, 'OPENAI_API_KEY', None) or os.environ.get('OPENAI_API_KEY', '')
    if not api_key or not str(api_key).strip().startswith('sk-'):
        raise ValueError(f"OpenAI API key not found or invalid")
    return OpenAI(api_key=api_key.strip())


def _encode_image_for_gpt(image_path, max_size=800):
    """Encode an image to base64 for GPT-4o, with size limit."""
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None, override_prompt=None):
    """
    Full Scene AI Reconstruction using GPT-4o + DALL-E 3.
    
    1. GPT-4o analyzes the room photo → text description
    2. GPT-4o analyzes the door product → text description  
    3. DALL-E 3 generates a new photorealistic room with the door installed
    """
    print(f"\n🎨 === DALL-E 3 VISUALIZATION START ===")
    
    client = _get_openai_client()
    
    door_name = product.name if hasattr(product, 'name') else 'Door'
    
    # Get door image path
    door_image_path = None
    if hasattr(product, 'image_no_bg') and product.image_no_bg and product.image_no_bg.name:
        try:
            if os.path.exists(product.image_no_bg.path):
                door_image_path = product.image_no_bg.path
        except:
            pass
    if not door_image_path and hasattr(product, 'original_image') and product.original_image and product.original_image.name:
        try:
            if os.path.exists(product.original_image.path):
                door_image_path = product.original_image.path
        except:
            pass
    if not door_image_path and hasattr(product, 'image') and product.image and product.image.name:
        try:
            if os.path.exists(product.image.path):
                door_image_path = product.image.path
        except:
            pass
    
    print(f"🚪 Door: {door_name}")
    print(f"📸 Room: {room_image_path}")
    print(f"🖼️ Door Image: {door_image_path or 'N/A'}")
    
    # --- Step 1: GPT-4o analyzes the room ---
    house_desc = "modern house entrance with light walls and wooden floor"
    try:
        print("📝 Step 1: GPT-4o analyzing room...")
        base64_room = _encode_image_for_gpt(room_image_path)
        r = client.chat.completions.create(
            model='gpt-4o',
            messages=[{
                'role': 'user',
                'content': [
                    {"type": "text", "text": "Describe this room/house entrance in detail: wall color, floor type, ceiling, curtains, furniture, lighting. Max 80 words. English only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_room}", "detail": "low"}}
                ]
            }],
            max_tokens=150
        )
        house_desc = r.choices[0].message.content
        print(f"   Room description: {house_desc[:100]}...")
    except Exception as e:
        print(f"   ⚠️ GPT-4o room analysis failed (using default): {e}")
    
    # --- Step 2: GPT-4o analyzes the door ---
    door_desc = door_name
    if door_image_path:
        try:
            print("📝 Step 2: GPT-4o analyzing door...")
            base64_door = _encode_image_for_gpt(door_image_path)
            r = client.chat.completions.create(
                model='gpt-4o',
                messages=[{
                    'role': 'user',
                    'content': [
                        {"type": "text", "text": "Describe this door in detail: material, color, pattern, handle style, panel design. Max 50 words. English only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_door}", "detail": "low"}}
                    ]
                }],
                max_tokens=120
            )
            door_desc = r.choices[0].message.content
            print(f"   Door description: {door_desc[:100]}...")
        except Exception as e:
            print(f"   ⚠️ GPT-4o door analysis failed (using name): {e}")
    
    # Professional Prompt Engineering (User's Methodology)
    prompt = (
        f"A highly realistic architectural visualization. "
        f"CONTEXT: A real-world room with these details: {house_desc}. "
        f"TASK: Replace the existing old door with this new double door design: {door_desc}. "
        f"PLACEMENT: Place the door base STRICTLY ON THE FLOOR LINE. It must be flush with the floor. "
        f"INTEGRATION: Perform seamless architectural integration into the existing wall. "
        f"LIGHTING: Match the natural lighting and shadows of the room exactly. "
        f"PRESERVATION: Maintain the original wall texture, floor, and room context 1:1. "
        f"Result: Photorealistic, 8k, professional interior design photograph."
    )
    
    print(f"   Prompt: {prompt[:150]}...")
    
    res = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size='1024x1024',
        quality='hd',
        n=1
    )
    
    url_result = res.data[0].url
    print(f"   ✅ DALL-E 3 image URL received!")
    
    # Download and save
    img_data = requests.get(url_result).content
    with open(result_image_path, 'wb') as f:
        f.write(img_data)
    
    print(f"   ✅ Saved to: {result_image_path}")
    print(f"🎨 === DALL-E 3 VISUALIZATION COMPLETE ===\n")
    
    return result_image_path
