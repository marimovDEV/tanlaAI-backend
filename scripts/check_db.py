import os
import sys
import django
from pathlib import Path

# Set up Django
sys.path.append(os.getcwd())
os.environ.setdemessagefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product, Category, Company

def run_diagnostics():
    print("--- 🔍 SYSTEM DIAGNOSTICS ---")
    print(f"Current Directory: {os.getcwd()}")
    
    # Check for all sqlite files
    sqlite_files = list(Path('.').rglob('*.sqlite3'))
    print(f"Databases found: {[str(p) for p in sqlite_files]}")
    
    # Current DB stats
    try:
        print(f"\n--- 📊 CURRENT DATABASE STATS ---")
        print(f"Categories: {Category.objects.count()}")
        print(f"Products: {Product.objects.count()}")
        print(f"Companies: {Company.objects.count()}")
        
        if Product.objects.exists():
            print(f"\n--- 🖼️ IMAGE PATH CHECK (First 5 Products) ---")
            for p in Product.objects.all()[:5]:
                print(f"Product: {p.name}")
                if p.image:
                    full_path = p.image.path
                    exists = os.path.exists(full_path)
                    print(f"  URL: {p.image.url}")
                    print(f"  Path: {p.image.name}")
                    print(f"  Exists on disk: {exists}")
                    if not exists:
                        # Check if it exists with one less 'products/'
                        alternative = full_path.replace('products/products/', 'products/')
                        if os.path.exists(alternative):
                            print(f"  💡 FOUND AT ALT PATH: {alternative}")
                else:
                    print(f"  Image: None")
    except Exception as e:
        print(f"Error during DB check: {e}")

if __name__ == "__main__":
    run_diagnostics()
