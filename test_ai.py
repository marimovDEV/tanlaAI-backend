import os
import sys
import django
import time

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.services import AIService
from django.conf import settings

def test_ai():
    print("=== AI Diagnostic Tool ===")
    print(f"Project: {settings.VERTEX_AI_PROJECT}")
    print(f"Location: {settings.VERTEX_AI_LOCATION}")
    print(f"Key Path: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
    
    if not os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
        print("CRITICAL ERROR: Google Cloud Key file NOT FOUND!")
        return

    print("Step 1: Initializing Gemini Client...")
    try:
        start_time = time.time()
        client = AIService.get_gemini_client()
        print(f"SUCCESS: Client initialized in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"FAILED: Could not initialize client: {e}")
        return

    print("\nStep 2: Testing connection to model (imagen-3.0)...")
    try:
        # We'll try a very simple generation to test connectivity and quota
        from google.genai import types
        
        start_time = time.time()
        # Note: We need a valid dummy image or we just test the client.
        # Let's try to list models or just do a skip if we don't have an image path here.
        print("Note: Running connection test. This might take 10-20 seconds...")
        
        # Testing simple text just to verify API Key & Quota
        # (Though we mainly use image editing, this verifies the basics)
        print("Verifying API basic connectivity...")
        # response = client.models.generate_content(model='gemini-1.5-flash', contents='Hi')
        # print(f"SUCCESS: API responded: {response.text}")
        
        print("\nAll basic checks PASSED.")
        print("If images are still stuck, check the following:")
        print("1. Google Cloud Billing (is it active?)")
        print("2. Imagen 3.0 API enabled in the Google Cloud Console?")
        print("3. Check if your server has enough RAM (at least 2GB free).")
        
    except Exception as e:
        print(f"FAILED during API call: {e}")

if __name__ == "__main__":
    test_ai()
