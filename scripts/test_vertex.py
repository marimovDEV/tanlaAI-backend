import os
import sys

# Add project root to path for Django settings
sys.path.append('c:\\Users\\user\\Downloads\\stitch\\stitch')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

import django
django.setup()

from shop.ai_utils import get_gemini_client
from google.genai import types

def test_vertex_ai():
    try:
        client = get_gemini_client()
        print("Successfully initialized Vertex AI client.")
        
        # Test a very simple generation to check API access
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Say "Vertex AI is working!"'
        )
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error test Vertex AI: {e}")

if __name__ == "__main__":
    test_vertex_ai()
