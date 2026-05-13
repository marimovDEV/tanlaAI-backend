import os, sys, django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.services import AIService
from google import genai
from google.genai import types

def test_vertex():
    import json
    
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
        img = Image.new('RGB', (1024, 1024), color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()
        
        mask = Image.new('RGBA', (1024, 1024), color=(0,0,0,0))
        # Add white solid rectangle for mask
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 300, 300], fill=(255,255,255,255))
        mbuf = io.BytesIO()
        mask.save(mbuf, format='PNG')
        mask_bytes = mbuf.getvalue()
        
        response = client.models.edit_image(
            model='imagen-3.0-capability-001',
            prompt="A blank wall.",
            reference_images=[
                types.RawReferenceImage(
                    reference_image=types.Image(image_bytes=img_bytes),
                    reference_id=0
                )
            ],
            config=types.EditImageConfig(
                edit_mode='EDIT_MODE_INPAINT_REMOVAL',
                mask=types.Image(image_bytes=mask_bytes),
                number_of_images=1
            )
        )
        print("SUCCESS:", len(response.generated_images))
    except Exception as e:
        print("ERROR:", e)

test_vertex()
