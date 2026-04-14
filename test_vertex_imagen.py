import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.services import AIService
from google import genai
from google.genai import types

def test_vertex():
    import json
    
    # Force vertex initialization with the local key
    key_path = "google-cloud-key.json"
    if not os.path.exists(key_path):
        print("ERROR: Key not found")
        return
        
    with open(key_path, 'r') as f:
        credentials = json.load(f)
    print("Project ID:", credentials.get('project_id'))
    
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        client = genai.Client(vertexai=True, project=credentials.get('project_id'), location='us-central1')
        print("Vertex AI Client built successfully!")
        
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt="Isolate the center object.",
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=img_bytes),
                    reference_id=0
                )
            ],
            config=types.EditImageConfig(
                edit_mode='EDIT_MODE_INPAINT_REMOVAL',
                number_of_images=1
            )
        )
        print("SUCCESS:", len(response.generated_images))
    except Exception as e:
        print("ERROR:", e)

test_vertex()
