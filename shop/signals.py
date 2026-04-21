from django.db import transaction
import threading
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Product, LeadRequest, Company
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


@receiver(post_save, sender=Company)
def notify_admin_new_company(sender, instance, created, **kwargs):
    """
    Sends a Telegram notification to the admin when a new company is created.
    """
    if created:
        from .notifications import NotificationService

        owner = instance.user
        owner_name = (
            f"{owner.first_name or ''} {owner.last_name or ''}".strip()
            if owner else "?"
        )
        username = f"@{owner.username}" if owner and owner.username else "—"
        phone = owner.phone if owner and owner.phone else "—"

        message = (
            "🏢 <b>YANGI KAMPANIYA YARATILDI</b>\n\n"
            f"📝 <b>Nomi:</b> {instance.name}\n"
            f"👤 <b>Egasi:</b> {owner_name}\n"
            f"📱 <b>Username:</b> {username}\n"
            f"📞 <b>Telefon:</b> {phone}\n\n"
            "🟡 <b>Status:</b> To'lov kutilmoqda\n\n"
            "❗ <i>Foydalanuvchi to'lov qilishini kutib turing.</i>"
        )

        thread = threading.Thread(
            target=NotificationService.send_telegram_message,
            args=(message,)
        )
        thread.start()
