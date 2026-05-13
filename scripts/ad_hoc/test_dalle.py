import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

import openai
from django.conf import settings

def test_dalle():
    client = openai.OpenAI(api_key=settings.OPEN_AI_KEY) # Wait, it might be OPENAI_API_KEY
    try:
        print("Testing DALL-E 3...")
        response = client.images.generate(
            model="dall-e-3",
            prompt="A sample room with a white door, professional architectural visualization.",
            n=1,
            size="1024x1024"
        )
        print(f"  SUCCESS! DALL-E generated image: {response.data[0].url[:100]}...")
    except Exception as e:
        print(f"  FAILED: {e}")

test_dalle()
