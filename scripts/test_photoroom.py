import requests
import os
import sys

# Photoroom API test script
API_KEY = "sk_pr_default_7620684dc06a0b0d05a41abab749c1618e67967c"
IMAGE_PATH = "/Users/ogabek/Documents/projects/tanlaAI/backend/media/products/door.jpg" # Example path

def test_photoroom():
    print(f"DEBUG: [Test] Testing Photoroom API with key {API_KEY[:10]}...")
    
    # Try to find an actual image file in media to test
    media_dir = "/Users/ogabek/Documents/projects/tanlaAI/backend/media/products"
    target_file = None
    if os.path.exists(media_dir):
        files = [f for f in os.listdir(media_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            target_file = os.path.join(media_dir, files[0])
            
    if not target_file:
        print("ERROR: No image found in media folder to test with.")
        return

    print(f"DEBUG: [Test] Sending file: {target_file}")
    
    url = "https://sdk.photoroom.com/v1/segment"
    try:
        with open(target_file, "rb") as f:
            files = {"image_file": f}
            headers = {"x-api-key": API_KEY}
            response = requests.post(url, files=files, headers=headers)
            
        if response.status_code == 200:
            print("SUCCESS! Photoroom API returned 200 OK.")
            print(f"Result size: {len(response.content)} bytes.")
            # Save a preview
            with open("photoroom_test_result.png", "wb") as out:
                out.write(response.content)
            print("Test result saved to photoroom_test_result.png")
        else:
            print(f"FAILED: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_photoroom()
