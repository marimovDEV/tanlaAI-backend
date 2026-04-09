import os
import sys
import django
from io import BytesIO
from PIL import Image
from django.core.files.base import ContentFile

# Set up Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product, Category, Company, AIResult

def optimize_image_field(instance, field_name, max_size=(1200, 1200)):
    field = getattr(instance, field_name)
    if not field:
        return False
    
    # Check if already optimized (webp)
    if field.name.endswith('.webp'):
        return False
        
    try:
        # Open the image
        img = Image.open(field.path)
        
        # Convert to RGB
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
            
        # Resize proportionaly if it exceeds max_size
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        # Save into buffer in WebP format
        output = BytesIO()
        img.save(output, format='WebP', quality=80, method=6)
        output.seek(0)
        
        # Construct the new filename with .webp extension
        base_name = os.path.splitext(field.name)[0]
        new_filename = f"{base_name}.webp"
        
        # Save back to the field
        # Note: we use save=True to commit to DB immediately
        field.save(new_filename, ContentFile(output.read()), save=True)
        return True
        
    except Exception as e:
        print(f"Error optimizing {field.name}: {e}")
        return False

def run_optimization():
    print(f"🚀 Starting global image optimization...")
    print(f"DEBUG: Current directory: {os.getcwd()}")
    print(f"DEBUG: SQLite Database check: {os.path.exists('db.sqlite3')}")

    # 1. Products
    print(f"📦 Optimizing Product images... (Total in DB: {Product.objects.count()})")
    p_count = 0
    for p in Product.objects.all():
        if p.image:
            print(f"  - Checking product: {p.name} | Image: {p.image.name}")
        if optimize_image_field(p, 'image'):
            p_count += 1
    print(f"✅ Optimized {p_count} product images.")

    # 2. Categories
    print("📂 Optimizing Category icons...")
    cat_count = 0
    for cat in Category.objects.all():
        if optimize_image_field(cat, 'icon'):
            cat_count += 1
    print(f"✅ Optimized {cat_count} category icons.")

    # 3. Companies
    print("🏢 Optimizing Company logos...")
    comp_count = 0
    for comp in Company.objects.all():
        if optimize_image_field(comp, 'logo'):
            comp_count += 1
    print(f"✅ Optimized {comp_count} company logos.")

    # 4. AI Results
    print("🤖 Optimizing AI Result images...")
    ai_count = 0
    for ai in AIResult.objects.all():
        if optimize_image_field(ai, 'image'):
            ai_count += 1
    print(f"✅ Optimized {ai_count} AI result images.")

    print("\n🏁 Optimization finished!")
    print(f"Total files processed: {p_count + cat_count + comp_count + ai_count}")

if __name__ == "__main__":
    run_optimization()
