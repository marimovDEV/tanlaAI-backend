from django.db import transaction
import threading
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Product, LeadRequest
from django.conf import settings


@receiver(post_save, sender=Product)
def trigger_ai_processing(sender, instance, created, **kwargs):
    """
    Triggers AI background removal when a product is created or image is updated.
    """
    auto_process_categories = [c.lower() for c in getattr(settings, 'AI_AUTO_PROCESS_CATEGORIES', [])]
    if (instance.category and instance.category.name.lower() in auto_process_categories 
        and instance.ai_status == 'none' and instance.image):
        
        def start_thread():
            from .services import AIService
            thread = threading.Thread(target=AIService.process_product_background, args=(instance,))
            thread.start()
            
        transaction.on_commit(start_thread)


@receiver(post_save, sender=LeadRequest)
def notify_new_lead_signal(sender, instance, created, **kwargs):
    """
    Sends a Telegram notification when a new lead is created.
    """
    if created:
        from .notifications import NotificationService
        
        # Run in a separate thread to avoid blocking the request
        thread = threading.Thread(target=NotificationService.notify_new_lead, args=(instance,))
        thread.start()
