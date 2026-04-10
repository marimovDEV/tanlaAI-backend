import os
import sys
import django
from pathlib import Path

# Setup Django environment
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.conf import settings
from google import genai
from google.genai import types

def get_client():
    api_key = getattr(settings, 'GEMINI_API_KEY', None)
    if api_key:
        print("DEBUG: Using API KEY for client.")
        return genai.Client(api_key=api_key)
    
    key_path = settings.GOOGLE_APPLICATION_CREDENTIALS
    if os.path.exists(key_path):
        print("DEBUG: Using Vertex AI (Service Account) for client.")
        return genai.Client(
            vertexai=True,
            project=settings.VERTEX_AI_PROJECT,
            location=settings.VERTEX_AI_LOCATION
        )
    return None

def test_gemini():
    print("--- Testing Gemini Text API ---")
    try:
        client = get_client()
        if not client: raise ValueError("No credentials found")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Hello, are you working?'
        )
        print(f"SUCCESS: Gemini responded: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"FAILED: Gemini error: {e}")
        return False

def test_imagen():
    print("\n--- Testing Imagen 3 Image Generation ---")
    try:
        client = get_client()
        if not client: raise ValueError("No credentials found")
        response = client.models.generate_image(
            model='imagen-3.0-generate-001',
            prompt='A simple wooden door',
            config=types.GenerateImageConfig(
                number_of_images=1,
                output_mime_type='image/png'
            )
        )
        if response.generated_images:
            print("SUCCESS: Imagen 3 generated an image successfully.")
            return True
        else:
            print("FAILED: Imagen 3 returned no images.")
            return False
    except Exception as e:
        print(f"FAILED: Imagen 3 error: {e}")
        return False

if __name__ == "__main__":
    g_ok = test_gemini()
    i_ok = test_imagen()
    
    print("\n--- Final Status ---")
    print(f"Gemini: {'✅ OK' if g_ok else '❌ FAILED'}")
    print(f"Imagen: {'✅ OK' if i_ok else '❌ FAILED'}")
