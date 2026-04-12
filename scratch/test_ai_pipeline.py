import os
import django
import sys

# Setup Django
sys.path.append('/Users/ogabek/Documents/projects/tanlaAI/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tanlaAI.settings')
django.setup()

from shop.models import Product
from shop.ai_utils import visualize_door_in_room
from PIL import Image

def run_test():
    print("--- STARTING LOCAL AI PIPELINE TEST ---")
    
    # 1. Get a sample product
    product = Product.objects.first()
    if not product:
        print("No products in DB. Please add one.")
        return

    # 2. Mock paths
    room_path = "/Users/ogabek/Documents/projects/tanlaAI/backend/media/temp_room.jpg"
    result_path = "/Users/ogabek/Documents/projects/tanlaAI/backend/media/test_result.jpg"
    
    # 3. Create dummy room if not exists
    if not os.path.exists(room_path):
        dummy_room = Image.new('RGB', (1024, 768), color=(200, 200, 200))
        dummy_room.save(room_path)
        print(f"Created dummy room at {room_path}")

    # 4. Run Visualization
    try:
        print(f"Testing visualization for product: {product.title}")
        output = visualize_door_in_room(product, room_path, result_path)
        print(f"SUCCESS! Output saved at: {output}")
    except Exception as e:
        print(f"FAILED! Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
