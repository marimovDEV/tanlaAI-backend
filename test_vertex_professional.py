import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from google import genai
from google.genai import types
from PIL import Image
import io
import json

def test_vertex():
    key_path = "google-cloud-key.json"
    if not os.path.exists(key_path):
        print("ERROR: Vertex AI Key not found")
        return

    with open(key_path, 'r') as f:
        credentials = json.load(f)
    
    project = credentials.get('project_id')
    print(f"Testing Vertex AI on project: {project}...")

    # Initialize Vertex AI client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
    client = genai.Client(vertexai=True, project=project, location='us-central1')

    room = Image.new('RGB', (1024, 1024), color=(204, 184, 163))
    buf = io.BytesIO()
    room.save(buf, format='PNG')
    room_bytes = buf.getvalue()

    door = Image.new('RGB', (400, 800), color=(240, 230, 221))
    buf2 = io.BytesIO()
    door.save(buf2, format='PNG')
    door_bytes = buf2.getvalue()

    try:
        # Vertex AI uses generate_content for Gemini models too
        # gemini-1.5-pro or gemini-1.5-flash are standard in Vertex
        model = 'gemini-1.5-pro-002' 
        print(f"Trying {model} on Vertex AI...")
        
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=room_bytes, mime_type='image/png'),
                types.Part.from_bytes(data=door_bytes, mime_type='image/png'),
                "Professional architectural visualization: Replace the door in the first image with the second door. Maintain exact lighting."
            ],
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    print(f"  SUCCESS! Vertex AI generated a Professional Rasm.")
                    return
        print("  Model responded but no image found (Vertex AI might need Imagen 3 enabled).")

    except Exception as e:
        print(f"  FAILED: {e}")

test_vertex()
