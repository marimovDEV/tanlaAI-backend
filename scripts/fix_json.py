import json

key_data = {
  "type": "service_account",
  "project_id": "ai-image-editor-492616",
  "private_key_id": "bd22f0dd29d3956bd56ddab140658d62c1658fbc",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCsU73mmzaBDUwn\nWR4EoXJ55CtBY/JDtE9A5IZRvqVDmoURhPswr68R5x+UyGuFwTge8S4GZG/UET/g\nk7ygJb9kzVqxIS7hqFL06aEvVsv3318w7nfSd8lqhNWfvt2ZEGKuOQXceXJXTeI1\nCtdN8yeTq1mn6b7Eily4wSBGptWP8uvEZ8PkV3pMBNwsJ26iEILglKUKrjtaqVC9\nUbDRD9NHaLUofMoi5fiqSqOCQTOTYNDHP8UqN4aJCFo9iANjGYg5fM3d5lB+zt+L\nTmAg2c7XBxwXAUk5LS9FUFq39UHv2yE/eBhT2nuQGRanCuBkwWfJmPB3j0SSHQAh\nQR/BIVARAgMBAAECggEAKd3xj+6LPCjRJOkpmhnoUFVfLfSclhbfP2aV/JO7HpTS\nugorJe209L5IhbL4rocePaQ+WJa7tZPYR1VVWTo6iQC8RTcI5xjkM+uT+caWaPsX\nqzFwoo1wPaUWqogqWebpdqcLdcB6x1u4dscSvpEx/xY9AsbPDnyIIOnoz8l2LtHB\nhiJPVfES7xoLlLeM8a1gJ5mKMPuhL2vgLVK0v557JJsq+LCdaPVS6VGwcFjqJKeM\nw++FwZW19XX+H2BoTIqdx8eIDullhdHUnKSo4/OSh8hTfRm91YVsIncKH6/GVInj\nQOq8fDSSQ30wj+dKSMNnKAzUZo3JEYlGsSH2oMhfXQKBgQDc6/dOfvX1CvAJq2Q4\nyMgmLfuASs8lixeW/TtoKNTeZDdS/OYB9smCCr9DEoBxOeHA7X6PcJxKg1HM+XsA\nttNfzqcMCPwebKmIXPVWYgbh0BvIu9JxOhfELqvye5vZJTc3mR5C9Q3lmb7cZnp5\nzJdE0t1FNjW4QY1tRpRSFswbdQKBgQDHsH/3Z1wpFFoxRGxYfUxuvNpaDVnNhtjN\nKE+2uAHd8lRZH+UQkinRf5+r04g8liPS5xU4EhnXS0m3tc2YcQkZm2XiVaUlJQUZ\n2fgswfzm86sBNMbTy7na2W4S35FpyIroELMqRpzbvYyWYuznPBHwtl1QtIKrg12/\ndNZxCQl6rQKBgHKXix4SSO4kEEJTvpadvwPe0hfHtg8ZSNEu5UOv+kqo3PGU8JGQ\ OxHTFUZrMGiKx4jVJ9KrbMZRu3qA5caHDrkhbhCWEICPiJiM700xZ7R1yBOlKRFG\nOtGuC86pQzutTCjwqXu9tMmlqBSWq2zGLKisX83owpCioANQmtrBrHmJAoGAe603\nNHXRwKeYTNdB+3RvOE7DNe765a5U7IvBAzvn/BywXRrB9odwWw6eR/+Va2DaAy+Y\nFsgvNQauO5fgJEAuEKwMaCf/RogtJpu3d5EWH7xe9zpGwrp0+7Sa1hmdqFTKo3xk\nWxSs5fP59NKEQ5sSyXFJTkjefXe5QnEMt6mPM2kCgYEAqm64NdXnk9JA1SniWXlx\nhQl6VVxfbyEvlphbwCLOfBpiBE2JlP/3WXmiSeXXg6W5aXL9uQdf022LggnnWHpP\nJxZ0xxhujONfs+6E8wzZJX22ETh237Y2cuW6r0Wjg6L4UdreJ7tCKLse73xBw7mu\n3/KnOFfsMrSHQYICfCZOGT4=\n-----END PRIVATE KEY-----\n",
  "client_email": "door-ai-editor@ai-image-editor-492616.iam.gserviceaccount.com",
  "client_id": "118162236460918197057",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/door-ai-editor%40ai-image-editor-492616.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

with open('google-cloud-key.json', 'w') as f:
    json.dump(key_data, f, indent=2)
print("Updated google-cloud-key.json")
