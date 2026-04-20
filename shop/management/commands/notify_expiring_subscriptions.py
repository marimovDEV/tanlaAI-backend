"""
Daily cron: warn company owners before their subscription expires.

    python manage.py notify_expiring_subscriptions

Sends a Telegram reminder to owners whose subscription_deadline is 3 or 1 days
away (configurable with --days). Intended to run once a day.

Not perfectly idempotent — if you run it twice on the same day, the owner gets
two messages. In production, schedule it exactly once (e.g. cron at 09:00).
If double-fire becomes a problem, we can persist the last-notified date on
Company and skip repeats.
"""
from django.core.management.base import BaseCommand
from django.utils import timezone

from shop.models import Company
from shop.notifications import NotificationService


class Command(BaseCommand):
    help = "Notify company owners whose subscription expires in N days."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            nargs="+",
            type=int,
            default=[3, 1],
            help="List of day offsets to notify on (default: 3 and 1 days before).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be sent without calling Telegram.",
        )

    def handle(self, *args, **options):
        offsets = options["days"]
        dry_run = options["dry_run"]
        now = timezone.now()

        total = 0
        for days in offsets:
            # Windows matched by calendar day: deadline falls between
            # [now + (days-1) days, now + days days]. This catches everyone
            # whose countdown currently reads `days` in whole days.
            low = now + timezone.timedelta(days=days - 1)
            high = now + timezone.timedelta(days=days)
            expiring = Company.objects.filter(
                subscription_deadline__gte=low,
                subscription_deadline__lt=high,
                is_active=True,
            ).select_related("user")

            for company in expiring:
                total += 1
                self.stdout.write(
                    f"  [{days}d] {company.name} (id={company.id}) "
                    f"→ owner {getattr(company.user, 'telegram_id', '?')}"
                )
                if not dry_run:
                    NotificationService.notify_subscription_expiring(company, days)

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"[DRY-RUN] Would notify {total} company owner(s)."
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"Notified {total} company owner(s)."
            ))
