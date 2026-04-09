import os
import io
import json
from google.genai import types
from google import genai

# Raw key with literal tags
raw_key = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCsU73mmzaBDUwn
WR4EoXJ55CtBY/JDtE9A5IZRvqVDmoURhPswr68R5x+UyGuFwTge8S4GZG/UET/g
k7ygJb9kzVqxIS7hqFL06aEvVsv3318w7nfSd8lqhNWfvt2ZEGKuOQXceXJXTeI1
CtdN8yeTq1mn6b7Eily4wSBGptWP8uvEZ8PkV3pMBNwsJ26iEILglKUKrjtaqVC9
UbDRD9NHaLUofMoi5fiqSqOCQTOTYNDHP8UqN4aJCFo9iANjGYg5fM3d5lB+zt+L
TmAg2c7XBxwXAUk5LS9FUFq39UHv2yE/eBhT2nuQGRanCuBkwWfJmPB3j0SSHQAh
QR/BIVARAgMBAAECggEAKd3xj+6LPCjRJOkpmhnoUFVfLfSclhbfP2aV/JO7HpTS
ugorJe209L5IhbL4rocePaQ+WJa7tZPYR1VVWTo6iQC8RTcI5xjkM+uT+caWaPsX
qzFwoo1wPaUWqogqWebpdqcLdcB6x1u4dscSvpEx/xY9AsbPDnyIIOnoz8l2LtHB
hiJPVfES7xoLlLeM8a1gJ5mKMPuhL2vgLVK0v557JJsq+LCdaPVS6VGwcFjqJKeM
w++FwZW19XX+H2BoTIqdx8eIDullhdHUnKSo4/OSh8hTfRm91YVsIncKH6/GVInj
QOq8fDSSQ30wj+dKSMNnKAzUZo3JEYlGsSH2oMhfXQKBgQDc6/dOfvX1CvAJq2Q4
yMgmLfuASs8lixeW/TtoKNTeZDdS/OYB9smCCr9DEoBxOeHA7X6PcJxKg1HM+XsA
ttNfzqcMCPwebKmIXPVWYgbh0BvIu9JxOhfELqvye5vZJTc3mR5C9Q3lmb7cZnp5
zJdE0t1FNjW4QY1tRpRSFswbdQKBgQDHsH/3Z1wpFFoxRGxYfUxuvNpaDVnNhtjN
KE+2uAHd8lRZH+UQkinRf5+r04g8liPS5xU4EhnXS0m3tc2YcQkZm2XiVaUlJQUZ
2fgswfzm86sBNMbTy7na2W4S35FpyIroELMqRpzbvYyWYuznPBHwtl1QtIKrg12/
dNZxCQl6rQKBgHKXix4SSO4kEEJTvpadvwPe0hfHtg8ZSNEu5UOv+kqo3PGU8JGQ
OxHTFUZrMGiKx4jVJ9KrbMZRu3qA5caHDrkhbhCWEICPiJiM700xZ7R1yBOlKRFG
OtGuC86pQzutTCjwqXu9tMmlqBSWq2zGLKisX83owpCioANQmtrBrHmJAoGAe603
NHXRwKeYTNdB+3RvOE7DNe765a5U7IvBAzvn/BywXRrB9odwWw6eR/+Va2DaAy+Y
FsgvNQauO5fgJEAuEKwMaCf/RogtJpu3d5EWH7xe9zpGwrp0+7Sa1hmdqFTKo3xk
WxSs5fP59NKEQ5sSyXFJTkjefXe5QnEMt6mPM2kCgYEAqm64NdXnk9JA1SniWXlx
hQl6VVxfbyEvlphbwCLOfBpiBE2JlP/3WXmiSeXXg6W5aXL9uQdf022LggnnWHpP
JxZ0xxhujONfs+6E8wzZJX22ETh237Y2cuW6r0Wjg6L4UdreJ7tCKLse73xBw7mu
3/KnOFfsMrSHQYICfCZOGT4=
-----END PRIVATE KEY-----"""

# Manual construct to avoid JSON escaping hell
key_dict = {
  "type": "service_account",
  "project_id": "ai-image-editor-492616",
  "private_key_id": "bd22f0dd29d3956bd56ddab140658d62c1658fbc",
  "private_key": raw_key,
  "client_email": "door-ai-editor@ai-image-editor-492616.iam.gserviceaccount.com",
  "client_id": "118162236460918197057",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/door-ai-editor%40ai-image-editor-492616.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

with open('google-cloud-key.json', 'w') as f:
    json.dump(key_dict, f, indent=2)

print("Saved google-cloud-key.json with actual newlines in private_key.")

try:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath('google-cloud-key.json')
    client = genai.Client(vertexai=True, project='ai-image-editor-492616', location='us-central1')
    print("Initializing client...")
    res = client.models.generate_content(model='gemini-1.5-flash', contents='Hello Vertex AI')
    print(f"Success! Response: {res.text}")
except Exception as e:
    print(f"Final error check: {e}")
