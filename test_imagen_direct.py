import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from google import genai
from google.genai import types

def test_imagen():
    key = "AIzaSyDwo0aAUA8Yv3oac3YFvVteTe_2pTrSKyQ" # Key 2
    client = genai.Client(api_key=key)
    
    try:
        print("Testing Imagen 3.0 Generate...")
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt='A high quality photo of a luxury white door in a modern room.',
            config=types.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        print(f"  SUCCESS! Imagen generated {len(response.generated_images)} image(s).")
    except Exception as e:
        print(f"  FAILED: {e}")

test_imagen()
