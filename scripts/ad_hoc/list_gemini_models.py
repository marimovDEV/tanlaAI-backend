"""List all available Gemini models on Vertex AI."""
import os
import json
from google import genai

def get_vertex_client():
    key_path = "/Users/ogabek/Documents/projects/tanlaAI/backend/google-cloud-key.json"
    with open(key_path, 'r') as f:
        info = json.load(f)
    return genai.Client(
        vertexai=True,
        project=info['project_id'],
        location="us-central1"
    )

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ogabek/Documents/projects/tanlaAI/backend/google-cloud-key.json"
client = get_vertex_client()

print("=== AVAILABLE MODELS (Vertex AI) ===")
try:
    for m in client.models.list():
        print(f"  {m.name}")
except Exception as e:
    print(f"  ❌ Error listing models: {e}")
