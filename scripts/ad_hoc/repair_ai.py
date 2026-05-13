import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from shop.models import Product
from shop.services import AIService

def force_repair_all():
    print("=== AI Force Repair Script ===")
    
    # Reset products that are stuck in 'processing' or haven't been processed
    products = Product.objects.all()
    count = 0
    
    for product in products:
        print(f"Checking Product: {product.name} (Status: {product.ai_status})")
        
        # Process if it has an image (even if already 'completed', to apply quality fixes)
        if product.image:
            print(f"   -> Forcing AI background removal for: {product.name}...")
            # Reset status to none first to bypass the service's internal 'completed' check
            product.ai_status = 'none'
            product.save(update_fields=['ai_status'])
            
            AIService.process_product_background(product)
            count += 1
            
    print(f"\nFINISH: Total {count} products processed.")

if __name__ == "__main__":
    force_repair_all()
