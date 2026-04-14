import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.services import AIService
from google.genai import types

def test_gen():
    client = AIService.get_gemini_client()
    try:
        res = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt='A simple room with a white door, red carpet, realistic photo, no luxury.',
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type='image/png'
            )
        )
        print("SUCCESS:", len(res.generated_images))
    except Exception as e:
        print("ERROR:", e)

test_gen()
