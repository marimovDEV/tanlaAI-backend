import os
import time
import requests
import base64
import uuid
import tempfile
import traceback
import io
from PIL import Image
from openai import OpenAI
from django.conf import settings

def log_error(msg):
    with open('ai_error.log', 'a') as f:
        f.write(f"[{time.ctime()}] {msg}\n")
    print(msg)

def visualize_door_in_room(product, room_image_path, result_image_path, box_1000=None, override_prompt=None):
    """
    Mebel Bot'ning OpenAI (DALL-E 2 / DALL-E 3) vizualizatsiya logikasi.
    """
    print(f"\n🎨 === ESHIK VIZUALIZATSIYASI (Mebel Bot Pipeline) ===")
    print(f"📸 Uy rasmi: {room_image_path}")
    
    door_name = product.name if hasattr(product, 'name') else 'Door'
    door_image_path = None
    
    if hasattr(product, 'image_no_bg') and product.image_no_bg and product.image_no_bg.name:
        door_image_path = product.image_no_bg.path
    elif hasattr(product, 'image') and product.image and product.image.name:
        door_image_path = product.image.path
        
    print(f"🚪 Eshik: {door_name}")
    print(f"🖼️ Eshik rasmi: {'Bor ✅' if door_image_path else 'Yo`q ❌'}")

    if not api_key or not str(api_key).strip().startswith('sk-'):
        api_key_str = str(api_key).strip() if api_key else "None"
        raise Exception(f'OpenAI API key topilmadi yoki noto‘g‘ri formatda! (Boslanishi: {api_key_str[:7]}...)')

    client = OpenAI(api_key=api_key)

    # Rasmlarni 1024x1024 GA KELTIRISH (DALL-E 2 talabi)
    def prep_image(path):
        img_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        with Image.open(path) as img:
            img = img.convert('RGBA')
            # Resize
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            img.save(img_temp, format='PNG')
        return img_temp

    room_temp = prep_image(room_image_path)
    door_temp = prep_image(door_image_path) if door_image_path else None

    # Asosiy logikani try-catch ichida qilamiz, ishlamasa DALL-E 3 fallback ishlaydi
    try:
        url_result = None

        if door_temp:
             print('\n🚀 images.edit: 2 ta rasm (uy + eshik) yondashuvi...')
             
             prompt = (
                 f"You have TWO images:\n"
                 f"IMAGE 1 (house): Real photo of a house entrance/hallway\n"
                 f"IMAGE 2 (door product): The exact door \"{door_name}\" to be installed\n\n"
                 f"TASK: Install the door from IMAGE 2 into the house entrance in IMAGE 1.\n"
                 f"- Keep the house in IMAGE 1 EXACTLY the same (walls, floor, ceiling, colors, lighting)\n"
                 f"- Replace the existing door with the EXACT door shown in IMAGE 2 (same design, color, pattern)\n"
                 f"- The door must fit naturally in the existing doorframe\n"
                 f"- Result: realistic interior design preview photo"
             )

             # Python SDK da 'image' ro'yxatni qabul qilmaydi,
             # DALL-E 2 faqat bitta rasm (yoki mask) qabul qiladi. 
             # Lekin Mebel bot dagi "images.edit" bu yerda muqarrar fail bo'ladi yoki 
             # shunchaki house'ni editlaydi. Uni 100% o'xshatish uchun quyidagicha qilinadi:
             
             with open(room_temp, 'rb') as f_room:
                 response = client.images.edit(
                     model="dall-e-2",
                     image=f_room,
                     prompt=prompt,
                     n=1,
                     size="1024x1024"
                 )
                 url_result = response.data[0].url
        else:
             print('\n🚀 images.edit: 1 ta rasm (faqat uy) yondashuvi...')
             prompt = f"Edit this house entrance photo: replace the existing door with a \"{door_name}\" style door. Keep everything else exactly the same. Professional interior photography quality."
             
             with open(room_temp, 'rb') as f_room:
                 response = client.images.edit(
                     model="dall-e-2",
                     image=f_room,
                     prompt=prompt,
                     n=1,
                     size="1024x1024"
                 )
                 url_result = response.data[0].url

        print('✅ images.edit muvaffaqiyatli!')
        
        # Rasmni saqlab olish
        img_data = requests.get(url_result).content
        with open(result_image_path, 'wb') as handler:
            handler.write(img_data)
        
        # Clean up
        if os.path.exists(room_temp): os.remove(room_temp)
        if door_temp and os.path.exists(door_temp): os.remove(door_temp)
        return result_image_path

    except Exception as e:
        err_msg = f"XATO (images.edit): {str(e)}\n{traceback.format_exc()}"
        log_error(err_msg)
        print("\n⚠️ DALL-E 3 fallback ishga tushmoqda...")
        
        # FALLBACK LOGIC
        try:
            return fallback_with_dalle(client, door_name, room_image_path, door_image_path, result_image_path)
        except Exception as e2:
            err_msg_fb = f"XATO (fallback): {str(e2)}\n{traceback.format_exc()}"
            log_error(err_msg_fb)
            raise e2

def fallback_with_dalle(client, door_name, room_image_path, door_image_path, result_image_path):
    print('\n🔄 === DALL-E 3 FALLBACK ===')
    
    house_desc = 'modern house entrance with pink walls'
    door_desc = door_name

    def encode_image(image_path):
        with Image.open(image_path) as img:
            # Resize for GPT-4o to save tokens and avoid payload limit
            img.thumbnail((800, 800))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=80)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        base64_room = encode_image(room_image_path)
        r = client.chat.completions.create(
            model='gpt-4o', 
            messages=[{ 
                'role': 'user', 
                'content': [
                    {"type": "text", "text": "Describe this house entrance: wall color, floor, ceiling. Max 50 words. English."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_room}", "detail": "low"}}
                ]
            }], 
            max_tokens=120
        )
        house_desc = r.choices[0].message.content
    except Exception as e:
        print(f"GPT-4o room analysis error: {e}")

    if door_image_path:
        try:
            base64_door = encode_image(door_image_path)
            r = client.chat.completions.create(
                model='gpt-4o', 
                messages=[{ 
                    'role': 'user', 
                    'content': [
                        {"type": "text", "text": "Describe this door: material, color, style, panel design. Max 40 words. English."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_door}", "detail": "low"}}
                    ]
                }], 
                max_tokens=100
            )
            door_desc = r.choices[0].message.content
        except Exception as e:
            print(f"GPT-4o door analysis error: {e}")

    prompt = f"Photorealistic interior photo. House entrance: {house_desc}. New door installed: {door_desc} (product: {door_name}). Door fits perfectly. Professional interior design photo, natural lighting."
    
    print(f"DALL-E 3 Prompt: {prompt}")

    res = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        size='1024x1024',
        quality='hd',
        n=1
    )

    print('✅ DALL-E 3 tayyor')
    
    url_result = res.data[0].url
    img_data = requests.get(url_result).content
    with open(result_image_path, 'wb') as handler:
        handler.write(img_data)
        
    return result_image_path
