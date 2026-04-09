import os
import django
import sys
import json

# Setup Django
sys.path.append('/Users/ogabek/Documents/projects/tanlaAI/backend')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from rest_framework.test import APIClient
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

def test_admin_token_auth():
    print("DEBUG: Testing Admin Token Authentication...")
    
    # 1. Ensure staff user exists
    from django.conf import settings
    if 'testserver' not in settings.ALLOWED_HOSTS:
        settings.ALLOWED_HOSTS.append('testserver')
        
    user, created = User.objects.get_or_create(username='admin', defaults={'is_staff': True})
    if created:
        user.set_password('admin123')
        user.save()
    
    client = APIClient()
    
    # 2. Test Login (Token Generation)
    print("DEBUG: Calling /api/v1/admin/login/...")
    response = client.post('/api/v1/admin/login/', {'username': 'admin', 'password': 'admin123'})
    
    if response.status_code == 200:
        token_key = response.data.get('token')
        print(f"SUCCESS: Login successful! Token: {token_key}")
    else:
        print(f"ERROR: Login failed with status {response.status_code}: {response.json()}")
        return

    # 3. Test Authenticated Request (Categories)
    print("DEBUG: Calling /api/v1/admin/categories/ with Token...")
    client.credentials(HTTP_AUTHORIZATION=f'Token {token_key}')
    response = client.get('/api/v1/admin/categories/')
    
    if response.status_code == 200:
        print(f"SUCCESS: Categories fetched successfully! Count: {len(response.data)}")
    else:
        print(f"ERROR: Categories fetch failed with status {response.status_code}: {response.json()}")

if __name__ == "__main__":
    test_admin_token_auth()
