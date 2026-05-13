import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

import openai
from django.conf import settings

def test_openai():
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        print("Testing DALL-E 3 directly...")
        response = client.images.generate(
            model="dall-e-3",
            prompt="A ultra-realistic photo of a high-end white door in a luxury room.",
            n=1,
            size="1024x1024"
        )
        print(f"  SUCCESS! DALL-E is working. Image URL: {response.data[0].url[:50]}...")
    except Exception as e:
        print(f"  FAILED: {e}")

test_openai()
