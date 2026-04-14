import os
import sys
import django

sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.services import AIService

try:
    product = Product.objects.first()
    if not os.path.exists('dummy_room.png'):
        from PIL import Image
        img = Image.new('RGB', (800, 800), color=(200, 200, 200))
        img.save('dummy_room.png')
    res = AIService.generate_room_preview(product, 'dummy_room.png', 'res.png')
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
