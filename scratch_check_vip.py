import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Company

try:
    c = Company.objects.first()
    if c:
        print(f"Company: {c.name}")
        print(f"Has is_vip field: {hasattr(c, 'is_vip')}")
        if hasattr(c, 'is_vip'):
            print(f"is_vip value: {c.is_vip}")
            # Try to update
            old_val = c.is_vip
            c.is_vip = not old_val
            c.save(update_fields=['is_vip'])
            print(f"Successfully toggled is_vip to: {c.is_vip}")
            # Toggle back
            c.is_vip = old_val
            c.save(update_fields=['is_vip'])
    else:
        print("No companies found.")
except Exception as e:
    print(f"Error: {e}")
