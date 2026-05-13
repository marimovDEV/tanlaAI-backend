from google import genai

client = genai.Client(api_key="AIzaSyARA_dDh__hqSUp6jUF-J9dJtpsFDoJ7cw")

for model in client.models.list():
    name = model.name
    if 'flash' in name.lower() or 'image' in name.lower() or 'gemini-2' in name.lower():
        methods = getattr(model, 'supported_generation_methods', [])
        print(f"{name} -> {methods}")
