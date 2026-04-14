import os
import sys
import django

sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.ai_utils import visualize_door_in_room

print("Starting test...")
# Get the first product
try:
    product = Product.objects.first()
    if not product:
        print("No products found.")
        sys.exit(1)
        
    print(f"Testing with Product: {product.name}")
    
    # Create dummy room image if none exists
    if not os.path.exists('dummy_room.png'):
        from PIL import Image
        img = Image.new('RGB', (800, 800), color=(200, 200, 200))
        img.save('dummy_room.png')
        
    result = visualize_door_in_room(product, 'dummy_room.png', 'test_result.png')
    print(f"Result saved to {result}")
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()
