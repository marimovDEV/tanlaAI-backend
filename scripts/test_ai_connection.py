import os
import django
import sys

# Setup Django
sys.path.append('/Users/ogabek/Documents/projects/tanlaAI/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.conf import settings
from shop.services import AIService

def test_gemini_connection():
    print("DEBUG: Testing Gemini/Vertex AI connection...")
    print(f"DEBUG: Project: {settings.VERTEX_AI_PROJECT}")
    print(f"DEBUG: Credentials: {settings.GOOGLE_APPLICATION_CREDENTIALS}")
    
    try:
        client = AIService.get_gemini_client()
        # Test a simple text generation to verify API access
        # Using a specific versioned name which is often more reliable in Vertex AI
        response = client.models.generate_content(
            model='gemini-1.5-flash-001',
            contents='Verify connection: say "Gemini is online"'
        )
        print(f"DEBUG: Response: {response.text}")
        print("SUCCESS: Gemini API is connected and working!")
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")

if __name__ == "__main__":
    test_gemini_connection()
