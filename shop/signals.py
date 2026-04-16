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
    Sends a Telegram notification when a new lead is created and sets a reminder.
    """
    if created:
        from .notifications import NotificationService
        
        # 1. Immediate Notification
        thread = threading.Thread(target=NotificationService.notify_new_lead, args=(instance,))
        thread.start()

        # 2. 10-Minute Reminder (Auto-Followup)
        def send_reminder():
            # Fresh fetch to check current status
            lead_id = instance.id
            try:
                from .models import LeadRequest
                current_lead = LeadRequest.objects.get(id=lead_id)
                if current_lead.status == 'new':
                    NotificationService.notify_lead_reminder(current_lead)
            except LeadRequest.DoesNotExist:
                pass

        reminder_timer = threading.Timer(600, send_reminder) # 600 seconds = 10 minutes
        reminder_timer.start()
