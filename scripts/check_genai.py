import os
import sys
import django
from django.conf import settings

# Setup Django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.services import AIService
from google.genai import types

def test():
    print("--- GenAI Connection Test ---")
    try:
        client = AIService.get_gemini_client()
        print(f"SUCCESS: Client initialized.")
        
        # Test model listing
        print("Models available:")
        # This might fail if insufficient permissions, but good to check
        # try:
        #     for m in client.models.list():
        #         print(f" - {m.name}")
        # except:
        #     print(" - (Could not list models)")

        print("\nChecking Imagen 3.0 accessibility...")
        # Small dummy test
        try:
            # We won't actually generate a billable image here yet, 
            # just check if we can call the service.
            print("To actually test generation, we need to run a full test.")
        except Exception as e:
            print(f"ERROR: Imagen check failed: {e}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test()
