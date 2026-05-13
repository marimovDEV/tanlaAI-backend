
import os
import django
import sys
import shutil

# Setup Django
sys.path.append("/Users/ogabek/Documents/projects/tanlaAI/backend")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from shop.models import Product, SystemSettings
from shop.services import AIService

class MockProduct:
    def __init__(self, id, name):
        self.id = id
        self.name = name

def test_integration():
    # Ensure provider is gemini_direct
    settings = SystemSettings.get_solo()
    settings.ai_provider = "gemini_direct"
    settings.save()
    
    # Mock a product since DB might be empty
    product = MockProduct(999, "Test Door")
    
    room_path = "/Users/ogabek/Documents/projects/tanlaAI/hona.png"
    door_path = "/Users/ogabek/Documents/projects/tanlaAI/eshik.png" # The root asset
    result_path = "/Users/ogabek/Documents/projects/tanlaAI/backend/final_integration_result.png"
    
    # We need to temporarily "hack" candidate_product_image_paths or just calling the direct method
    print(f"Testing integration for Mock product: {product.name}")
    
    # Mock image field
    class MockField:
        def __init__(self, path):
            self.name = "mock_img.png"
            self.path = path
            
    product.image = MockField(door_path)
    
    try:
        # We call the method directly to verify its internal logic works with real service account/keys
        result = AIService.generate_with_gemini_direct(product, room_path, result_path)
        print(f"Success! Result saved to: {result}")
    except Exception as e:
        print(f"Integration failed: {e}")

if __name__ == "__main__":
    test_integration()
