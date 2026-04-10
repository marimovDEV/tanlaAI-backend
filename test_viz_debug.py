import os
import sys
import django
import json

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.services import AIService

def test_visualization(room_image_path):
    print("=== Visualization Deep Debug ===")
    
    if not os.path.exists(room_image_path):
        print(f"ERROR: Room image not found: {room_image_path}")
        return

    # Pick a product that is already processed (completed)
    product = Product.objects.filter(ai_status='completed').first()
    if not product:
        print("ERROR: No completed products found to test with.")
        return

    print(f"Using Product: {product.name} (ID: {product.id})")
    print(f"Using Room Image: {room_image_path}")
    
    result_path = "media/ai_results/debug_test.png"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    try:
        # We manually call the service method to see the logs
        print("Calling AIService.generate_room_preview...")
        final_path = AIService.generate_room_preview(product, room_image_path, result_path)
        print(f"SUCCESS: Result saved to {final_path}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_viz_debug.py <path_to_room_image>")
        # Try a default if it exists
        default_room = "media/ai_temp/test_room.png"
        if os.path.exists(default_room):
            test_visualization(default_room)
        else:
            print("Please provide a room image path.")
    else:
        test_visualization(sys.argv[1])
