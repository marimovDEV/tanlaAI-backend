import os
import sys
import django
import io
from PIL import Image
import numpy as np

# Setup Django environment
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.conf import settings
from shop.services import AIService
from shop.models import Product

def test_hybrid_background_removal(image_path):
    """
    Test script to verify the OpenAI GPT-4o + SAM hybrid pipeline.
    """
    print(f"--- Starting Hybrid AI Test: OpenAI + SAM ---")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    # 1. Mock a product object for testing
    # We'll use the FIRST product in database as a dummy container
    product = Product.objects.first()
    if not product:
        print("ERROR: No products found in database to use as dummy.")
        return

    print(f"Using Product ID: {product.id} ({product.name}) as dummy container.")

    # 2. Manually overwrite the background removal process logic for this test run
    # OR better: just call the service and monitor results.
    # Note: process_product_background normally expects a Product object and saves to its fields.
    
    # Let's verify OpenAI API Key is present
    openai_key = getattr(settings, 'OPENAI_API_KEY', None)
    if not openai_key:
        print("WARNING: OPENAI_API_KEY not found in settings. Fallback to Gemini will occur.")
    else:
        print("SUCCESS: OpenAI API Key found.")

    try:
        # We'll temporarily point the product's image to our test image
        with open(image_path, 'rb') as f:
            from django.core.files.base import ContentFile
            product.original_image.save('test_input.jpg', ContentFile(f.read()), save=True)
        
        # Reset status to force re-processing
        product.ai_status = 'none'
        product.save()

        print("Triggering AIService.process_product_background...")
        AIService.process_product_background(product)
        
        product.refresh_from_db()
        if product.ai_status == 'completed':
            print(f"--- SUCCESS! ---")
            print(f"Isolated Image saved at: {product.image_no_bg.path}")
            print(f"Check this file to verify the quality.")
        else:
            print(f"FAILURE: AI Status is {product.ai_status}")

    except Exception as e:
        print(f"CRITICAL ERROR during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If no image provided, try to find one in media
    test_img = os.path.join(settings.MEDIA_ROOT, 'products', 'test_door.jpg')
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    
    test_hybrid_background_removal(test_img)
