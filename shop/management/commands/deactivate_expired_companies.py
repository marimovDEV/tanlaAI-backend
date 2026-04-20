"""
Daily cron: deactivate products for any company whose subscription has expired.

Run from crontab / systemd timer:
    python manage.py deactivate_expired_companies

Idempotent — running it twice is harmless. It only flips products that are
currently is_active=True, and only for companies whose subscription_deadline
is in the past. No effect on companies with subscription_deadline=NULL (those
are treated as "no paid subscription set yet" and left alone — the marketplace
listing filter already handles the deadline=None case separately).

Also sends a one-time Telegram notification to the company owner when their
products have just been deactivated, so they know why their listings vanished.
"""
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from shop.models import Company, Product
from shop.notifications import NotificationService


class Command(BaseCommand):
    help = "Deactivate products for companies whose subscription has expired."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would change without writing to the DB.",
        )
        parser.add_argument(
            "--notify",
            action="store_true",
            default=True,
            help="Send Telegram notification to affected company owners.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        notify = options["notify"]
        now = timezone.now()

        expired = Company.objects.filter(
            subscription_deadline__lt=now,
            subscription_deadline__isnull=False,
        ).select_related("user")

        total_companies = 0
        total_products = 0

        for company in expired:
            # Count products that will actually flip — this is the number we
            # report to the owner. Skip the company entirely if nothing would
            # change (their products are already paused).
            affected_qs = Product.objects.filter(company=company, is_active=True)
            count = affected_qs.count()
            if count == 0:
                continue

            total_companies += 1
            total_products += count

            self.stdout.write(
                f"  {company.name} (id={company.id}): {count} product(s) -> inactive"
            )

            if dry_run:
                continue

            with transaction.atomic():
                affected_qs.update(is_active=False)

            if notify and company.user and company.user.telegram_id:
                message = (
                    "⚠️ <b>Obuna muddati tugadi</b>\n\n"
                    f"Sizning <b>{company.name}</b> kompaniyangizning obuna "
                    f"muddati tugadi va <b>{count} ta mahsulot</b> vaqtinchalik "
                    "yashirildi.\n\n"
                    "Mahsulotlar yana ko'rinishi uchun obunani yangilang."
                )
                NotificationService.send_telegram_message(
                    message, chat_id=str(company.user.telegram_id)
                )

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"[DRY-RUN] Would deactivate {total_products} product(s) "
                f"across {total_companies} company(ies)."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Deactivated {total_products} product(s) across "
                f"{total_companies} company(ies)."
            ))
