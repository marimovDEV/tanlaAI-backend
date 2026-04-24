from django.utils import timezone
from datetime import timedelta
from django.db import transaction
from django.conf import settings
from .models import Payment, Company, Subscription, Product, SystemBilling, TelegramUser
from .notifications import NotificationService

class PaymentService:
    @staticmethod
    @transaction.atomic
    def approve_payment(payment, reviewed_by_tg_user=None):
        """
        Logic for approving a payment.
        Extends subscription, activates company and products.
        """
        if payment.status != "pending":
            return False, f"To'lov allaqachon {payment.get_status_display()} holatida."

        now = timezone.now()
        company = payment.company
        billing_config = SystemBilling.get_solo()
        days_per_month = billing_config.subscription_days or 30

        # Extend from LATER of (now, existing deadline)
        current_deadline = company.subscription_deadline
        base = current_deadline if current_deadline and current_deadline > now else now
        new_deadline = base + timedelta(days=days_per_month * payment.months)
        
        company.subscription_deadline = new_deadline
        company.is_active = True
        company.status = "active"
        company.save(update_fields=["subscription_deadline", "is_active", "status"])

        # Sync Subscription model
        sub, _ = Subscription.objects.get_or_create(company=company)
        sub.expires_at = new_deadline
        sub.save(update_fields=["expires_at"])

        # Reactivate products
        reactivated = Product.objects.filter(
            company=company, is_active=False
        ).update(is_active=True)

        payment.status = "approved"
        payment.reviewed_at = now
        if reviewed_by_tg_user:
            payment.reviewed_by = reviewed_by_tg_user
        payment.save(update_fields=["status", "reviewed_at", "reviewed_by"])

        # Notifications
        NotificationService.notify_payment_approved(payment, reactivated_count=reactivated)
        NotificationService.notify_admin_payment_approved(payment)
        
        return True, "To'lov muvaffaqiyatli tasdiqlandi."

    @staticmethod
    @transaction.atomic
    def reject_payment(payment, reason, reviewed_by_tg_user=None):
        """
        Logic for rejecting a payment.
        """
        if payment.status != "pending":
            return False, f"To'lov allaqachon {payment.get_status_display()} holatida."

        payment.status = "rejected"
        payment.rejection_reason = reason
        payment.reviewed_at = timezone.now()
        if reviewed_by_tg_user:
            payment.reviewed_by = reviewed_by_tg_user
        payment.save(update_fields=["status", "rejection_reason", "reviewed_at", "reviewed_by"])

        # Notification to user
        NotificationService.notify_payment_rejected(payment)
        
        return True, "To'lov rad etildi."
