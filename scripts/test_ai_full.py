#!/usr/bin/env python3
"""
=================================================================
TanlaAI — Serverda AI tizimlarini to'liq tekshirish skripti
=================================================================
Ishga tushirish:
    cd /path/to/backend
    source venv/bin/activate   (yoki venv_mac)
    python scripts/test_ai_full.py
=================================================================
"""
import os
import sys
import time
import json
import traceback
from pathlib import Path

# Django setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

import django
django.setup()

from django.conf import settings

PASS = "✅ OK"
FAIL = "❌ XATO"
WARN = "⚠️  OGOHLANTIRISH"

results = []

def log(icon, msg):
    print(f"  {icon}  {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ========== TEST 1: Settings tekshiruvi ==========
def test_settings():
    section("1. SOZLAMALAR TEKSHIRUVI")
    
    api_key = getattr(settings, 'GEMINI_API_KEY', '')
    if api_key:
        log(PASS, f"GEMINI_API_KEY: ...{api_key[-8:]}")
    else:
        log(WARN, "GEMINI_API_KEY: bo'sh (Service Account ishlatiladi)")
    
    key_path = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', '')
    if key_path and os.path.exists(key_path):
        log(PASS, f"Service Account fayli topildi: {key_path}")
        with open(key_path, 'r') as f:
            info = json.load(f)
        log(PASS, f"  Project ID: {info.get('project_id', 'NOANIQ')}")
        log(PASS, f"  Client Email: {info.get('client_email', 'NOANIQ')}")
        pk = info.get('private_key', '')
        if pk and len(pk) > 50:
            log(PASS, f"  Private Key: bor ({len(pk)} belgi)")
        else:
            log(FAIL, f"  Private Key: YO'Q yoki juda qisqa ({len(pk)} belgi)")
            results.append(("Service Account Private Key", False))
            return
    else:
        log(FAIL, f"Service Account fayli TOPILMADI: {key_path}")
        results.append(("Service Account fayli", False))
    
    project = getattr(settings, 'VERTEX_AI_PROJECT', '')
    location = getattr(settings, 'VERTEX_AI_LOCATION', '')
    log(PASS if project else FAIL, f"VERTEX_AI_PROJECT: {project or 'EMPTY'}")
    log(PASS if location else FAIL, f"VERTEX_AI_LOCATION: {location or 'EMPTY'}")
    results.append(("Sozlamalar", bool(api_key or (key_path and os.path.exists(key_path)))))


# ========== TEST 2: Client yaratish ==========
def test_client_creation():
    section("2. AI CLIENT YARATISH")
    try:
        from shop.services import AIService
        t0 = time.time()
        client = AIService.get_gemini_client()
        dt = time.time() - t0
        log(PASS, f"Client yaratildi ({dt:.2f}s): {type(client).__name__}")
        results.append(("Client yaratish", True))
        return client
    except Exception as e:
        log(FAIL, f"Client yaratib bo'lmadi: {e}")
        traceback.print_exc()
        results.append(("Client yaratish", False))
        return None


# ========== TEST 3: Gemini text ==========
def test_gemini_text(client):
    section("3. GEMINI TEXT TEST")
    if not client:
        log(FAIL, "Client yo'q, o'tkazib yuboryapman")
        results.append(("Gemini Text", False))
        return
    
    try:
        t0 = time.time()
        # Try multiple model names
        for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-pro']:
            try:
                log("🔄", f"Model sinash: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents='Say "Hello TanlaAI" in one line.'
                )
                dt = time.time() - t0
                log(PASS, f"Model ishladi: {model_name} ({dt:.2f}s)")
                log(PASS, f"Javob: {response.text.strip()[:100]}")
                results.append(("Gemini Text", True))
                return
            except Exception as e:
                log(WARN, f"  {model_name} ishlamadi: {str(e)[:80]}")
        
        log(FAIL, "Hech bir Gemini model ishlamadi")
        results.append(("Gemini Text", False))
    except Exception as e:
        log(FAIL, f"Gemini text xatolik: {e}")
        results.append(("Gemini Text", False))


# ========== TEST 4: Imagen 3 edit_image ==========
def test_imagen_edit(client):
    section("4. IMAGEN 3 EDIT_IMAGE TEST")
    if not client:
        log(FAIL, "Client yo'q, o'tkazib yuboryapman")
        results.append(("Imagen edit_image", False))
        return
    
    try:
        from google.genai import types
        from PIL import Image
        import io
        
        # Create simple test images
        log("🔄", "Test rasmlari yaratilmoqda...")
        room = Image.new("RGB", (512, 512), (220, 210, 200))  # Beige wall
        mask = Image.new("L", (512, 512), 0)
        # Draw white rectangle in center (mask area)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([180, 100, 330, 450], fill=255)
        
        r_buf = io.BytesIO()
        room.save(r_buf, format='PNG')
        m_buf = io.BytesIO()
        mask.save(m_buf, format='PNG')
        
        log("🔄", "Imagen 3 so'rov yuborilmoqda...")
        t0 = time.time()
        
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt='Place a simple wooden door in the masked area. The door should have a handle.',
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=r_buf.getvalue()),
                    reference_id=0,
                )
            ],
            config=types.EditImageConfig(
                edit_mode='INPAINT_INSERT',
                number_of_images=1,
                output_mime_type='image/png',
                mask=types.Image(image_bytes=m_buf.getvalue()),
            ),
        )
        dt = time.time() - t0
        
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            log(PASS, f"Imagen 3 natija berdi! ({dt:.2f}s, {len(img_bytes)} bytes)")
            
            # Save test result
            test_dir = os.path.join(settings.MEDIA_ROOT, 'ai_test')
            os.makedirs(test_dir, exist_ok=True)
            test_path = os.path.join(test_dir, 'imagen_test_result.png')
            with open(test_path, 'wb') as f:
                f.write(img_bytes)
            log(PASS, f"Test natija saqlandi: {test_path}")
            results.append(("Imagen edit_image", True))
        else:
            log(FAIL, f"Imagen 3 natija BERMADI ({dt:.2f}s)")
            results.append(("Imagen edit_image", False))
            
    except Exception as e:
        log(FAIL, f"Imagen 3 xatolik: {e}")
        traceback.print_exc()
        results.append(("Imagen edit_image", False))


# ========== TEST 5: Mahsulot assetlarini tekshirish ==========
def test_product_assets():
    section("5. MAHSULOT ASSETLARI TEKSHIRUVI")
    try:
        from shop.models import Product
        products = Product.objects.all()
        log("📦", f"Jami mahsulotlar: {products.count()}")
        
        for p in products:
            print(f"\n  --- #{p.id}: {p.name} (ai_status: {p.ai_status}) ---")
            for attr in ['image', 'image_no_bg', 'original_image']:
                field = getattr(p, attr, None)
                if field and field.name:
                    exists = os.path.exists(field.path)
                    if exists:
                        size = os.path.getsize(field.path) / 1024
                        from PIL import Image
                        try:
                            img = Image.open(field.path)
                            log(PASS, f"{attr}: {field.name} ({size:.0f}KB, {img.size}, {img.mode})")
                        except Exception as e:
                            log(WARN, f"{attr}: fayl bor lekin ochib bo'lmadi: {e}")
                    else:
                        log(FAIL, f"{attr}: fayl TOPILMADI: {field.path}")
                else:
                    log("⬜", f"{attr}: bo'sh")
        
        results.append(("Mahsulot assetlari", True))
    except Exception as e:
        log(FAIL, f"Assetlarni tekshirishda xato: {e}")
        results.append(("Mahsulot assetlari", False))


# ========== TEST 6: To'liq vizualizatsiya simulyatsiyasi ==========
def test_full_visualization(client):
    section("6. TO'LIQ VIZUALIZATSIYA SIMULYATSIYASI")
    if not client:
        log(FAIL, "Client yo'q")
        results.append(("Full Visualization", False))
        return
    
    try:
        from shop.models import Product
        from PIL import Image
        import io
        
        # Eshik kategoriyasidagi birinchi mahsulotni topish
        product = None
        for p in Product.objects.all():
            cat_name = p.category.name.lower() if p.category else ''
            if 'eshik' in cat_name or 'door' in cat_name:
                product = p
                break
        
        if not product:
            product = Product.objects.first()
        
        if not product:
            log(FAIL, "Hech qanday mahsulot yo'q")
            results.append(("Full Visualization", False))
            return
        
        log("📦", f"Test mahsulot: #{product.id} — {product.name}")
        
        # Create test room image
        test_dir = os.path.join(settings.MEDIA_ROOT, 'ai_test')
        os.makedirs(test_dir, exist_ok=True)
        
        room_path = os.path.join(test_dir, 'test_room.png')
        result_path = os.path.join(test_dir, 'test_result.png')
        
        room = Image.new("RGB", (800, 600), (230, 220, 210))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(room)
        # Draw a simple wall with doorframe
        draw.rectangle([300, 80, 500, 520], fill=(180, 170, 160))
        room.save(room_path)
        log(PASS, f"Test xona rasmi yaratildi: {room_path}")
        
        # Run visualization
        from shop.ai_utils import visualize_door_in_room
        log("🔄", "visualize_door_in_room ishga tushmoqda...")
        t0 = time.time()
        
        result = visualize_door_in_room(product, room_path, result_path)
        dt = time.time() - t0
        
        if result and os.path.exists(result_path):
            size = os.path.getsize(result_path) / 1024
            log(PASS, f"Natija tayyor! ({dt:.2f}s, {size:.0f}KB)")
            log(PASS, f"Fayl: {result_path}")
            results.append(("Full Visualization", True))
        else:
            log(FAIL, f"Natija fayli yo'q ({dt:.2f}s)")
            results.append(("Full Visualization", False))
            
    except Exception as e:
        log(FAIL, f"Vizualizatsiyada xatolik: {e}")
        traceback.print_exc()
        results.append(("Full Visualization", False))


# ========== MAIN ==========
if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("  TanlaAI — AI TIZIMLAR DIAGNOSTIKASI")
    print("🚀" * 30)
    
    test_settings()
    client = test_client_creation()
    test_gemini_text(client)
    test_imagen_edit(client)
    test_product_assets()
    test_full_visualization(client)
    
    section("YAKUNIY NATIJA")
    all_ok = True
    for name, ok in results:
        icon = PASS if ok else FAIL
        print(f"  {icon}  {name}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print(f"\n  🎉 BARCHA TESTLAR O'TDI!")
    else:
        print(f"\n  💥 BA'ZI TESTLAR O'TMADI — yuqoridagi xatolarni tekshiring.")
    print()
