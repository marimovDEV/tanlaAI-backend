import os
import django
import sys
import io
from PIL import Image as PILImage

# Setup Django
sys.path.append('/Users/ogabek/Documents/projects/tanlaAI/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.conf import settings
from shop.services import AIService
from google.genai import types

def test_imagen_functionality():
    print("DEBUG: Testing Imagen 3 (Inpainting) connection...")
    
    try:
        client = AIService.get_gemini_client()
        
        # Create a tiny dummy image
        img = PILImage.new('RGB', (100, 100), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        print("DEBUG: Calling Imagen 3 edit_image...")
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt="Isolate the center object.",
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=img_bytes),
                    reference_id=0
                )
            ],
            config=types.EditImageConfig(
                edit_mode='EDIT_MODE_INPAINT_REMOVAL',  # Full enum string
                number_of_images=1
            )
        )
        
        if response.generated_images:
            print("SUCCESS: Imagen 3 is online and ready!")
        else:
            print("WARNING: Imagen 3 call succeeded but no images returned.")
            
    except Exception as e:
        print(f"ERROR: Imagen 3 test failed: {e}")

if __name__ == "__main__":
    test_imagen_functionality()
