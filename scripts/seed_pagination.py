import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product, Category

def seed_more_products():
    category = Category.objects.first()
    if not category:
        category = Category.objects.create(name="General")
    
    for i in range(1, 41):
        Product.objects.create(
            name=f"Test Product {i}",
            description=f"Description for test product {i}",
            price=10.0 + i,
            category=category
        )
    print("Created 40 test products.")

if __name__ == "__main__":
    seed_more_products()
