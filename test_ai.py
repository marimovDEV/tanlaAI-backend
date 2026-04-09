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
    print("=== AI Diagnostic Tool (API KEY / JSON) ===")
    
    api_key = getattr(settings, 'GEMINI_API_KEY', '')
    key_path = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', '')
    
    print(f"API KEY: {'FOUND' if api_key else 'NOT FOUND'}")
    print(f"JSON Key Path: {key_path}")
    print(f"JSON Key Exists: {os.path.exists(key_path)}")

    print("\nStep 1: Initializing Client...")
    try:
        start_time = time.time()
        client = AIService.get_gemini_client()
        
        mode = "VERTEX AI" if getattr(client, 'vertexai', False) else "GOOGLE AI (API KEY)"
        print(f"SUCCESS: Client initialized in {mode} mode ({time.time() - start_time:.2f}s)")
        
        print("\nStep 2: Testing connection...")
        # A simple operation to test connectivity
        models = client.models.list_models()
        print("SUCCESS: Connection established. Models listed.")
        
        print("\nStep 3: Testing local background removal (rembg)...")
        import rembg
        import numpy as np
        # Simple test to see if it loads
        dummy_input = np.zeros((10, 10, 3), dtype=np.uint8)
        rembg.remove(dummy_input)
        print("SUCCESS: rembg is working locally.")
        
        print("\nALL SYSTEMS OPERATIONAL!")
        
    except Exception as e:
        print(f"FAILED: {e}")
        if not api_key:
            print("\nTIP: Please add GEMINI_API_KEY to your .env file to enable API Key mode.")

if __name__ == "__main__":
    test_ai()
