import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product

def delete_all_products():
    count = Product.objects.count()
    if count == 0:
        print("База данных уже пуста. Продукты не найдены.")
        return
    
    # Deleting all products
    Product.objects.all().delete()
    print(f"Успешно удалено {count} продуктов из базы данных.")

if __name__ == "__main__":
    delete_all_products()
