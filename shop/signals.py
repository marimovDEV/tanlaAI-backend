from django.db import transaction
import threading
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Product
from django.conf import settings


@receiver(post_save, sender=Product)
def trigger_ai_processing(sender, instance, created, **kwargs):
    """
    Triggers AI background removal when a product is created or image is updated,
    if the category belongs to the target categories (e.g., 'Eshiklar').
    Uses AIService for clean separation of concerns.
    """
    auto_process_categories = [c.lower() for c in getattr(settings, 'AI_AUTO_PROCESS_CATEGORIES', [])]
    if (instance.category and instance.category.name.lower() in auto_process_categories 
        and instance.ai_status == 'none' and instance.image):
        
        def start_thread():
            from .services import AIService
            thread = threading.Thread(target=AIService.process_product_background, args=(instance,))
            thread.start()
            
        transaction.on_commit(start_thread)
