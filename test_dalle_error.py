import sys, os, django

sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.ai_utils import visualize_door_in_room
import traceback

print("Testing DALL-E 3 manually to catch the exact error...")
try:
    product = Product.objects.filter(name__icontains="eshik").first()
    if not product:
        product = Product.objects.first()
    print(f"Product: {product.name}")
    
    # create a dummy image
    from PIL import Image
    if not os.path.exists('dummy.png'):
        img = Image.new('RGB', (800, 800), color=(100, 100, 100))
        img.save('dummy.png')
        
    visualize_door_in_room(product, 'dummy.png', 'res.png')
    print("SUCCESS.")
except Exception as e:
    print(f"\nCaught Exception: {type(e).__name__}: {e}")
