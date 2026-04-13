import os
import sys
import django
import cv2
import numpy as np

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.services import AIService
from shop.sam_utils import SAMService

def test_pipeline():
    print("=== Testing Door Perspective Pipeline ===")
    
    # 1. Get a test product (or first one)
    product = Product.objects.first()
    if not product:
        print("No products found in DB. Please run seed_db.py first.")
        return

    # 2. Use a test room image
    # Note: Replace with a real path to a room image for a real test
    room_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "ai_temp", "test_room.png")
    
    # If test_room doesn't exist, we can't do a full visual test, 
    # but we can check if dependencies load.
    if not os.path.exists(room_path):
        print(f"Test room '{room_path}' not found. Checking dependencies...")
        try:
            predictor = SAMService.get_predictor()
            print("SAM Predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM: {e}")
            return
        print("Dependency check passed.")
        return

    result_path = "test_visualization_result.png"
    
    try:
        AIService.generate_room_preview(product, room_path, result_path)
        print(f"Success! Result saved to {result_path}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
