"""
Daily cron: prune old AI visualization results so media storage stays bounded.

    python manage.py cleanup_ai_results              # delete older than 1 day
    python manage.py cleanup_ai_results --days 7     # keep a week
    python manage.py cleanup_ai_results --dry-run    # show what would be deleted
    python manage.py cleanup_ai_results --keep-lead-linked
        # skip AIResults attached to open (non-closed) leads

AIResults are ephemeral — the user viewed their room-preview image, and we
keep it for a short grace period so the Telegram message link doesn't 404 too
fast. After that, delete both the DB row AND the underlying image files so
disk doesn't fill up.

LeadRequest.ai_result is a SET_NULL FK, so deletion does NOT cascade to leads.
Leads just lose their ai_result pointer, which is harmless — the lead itself
is still visible with all its other fields.

Safety:
  - Deletion runs in small batches to avoid long-held row locks on big tables.
  - File deletion is best-effort: a missing file on disk is logged but doesn't
    stop the DB delete. We never raise — a failed cleanup tomorrow is fine.
  - With --keep-lead-linked we preserve AIResults for any open sale, so the
    company owner still has visual context when they contact the customer.
"""
import os

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from shop.models import AIResult


# Lead statuses that mean "deal is still in flight" — keep AI evidence for these.
OPEN_LEAD_STATUSES = {"new", "contacted", "active"}


class Command(BaseCommand):
    help = "Delete AIResult rows (and their image files) older than N days."

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=1,
            help="Delete AIResults older than this many days (default: 1).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would be deleted without touching the DB or disk.",
        )
        parser.add_argument(
            "--keep-lead-linked",
            action="store_true",
            help="Skip AIResults linked to leads that aren't closed yet.",
        )
        parser.add_argument(
            "--batch",
            type=int,
            default=500,
            help="Rows per batch to avoid long-running transactions (default: 500).",
        )

    def handle(self, *args, **options):
        days = options["days"]
        dry_run = options["dry_run"]
        keep_linked = options["keep_lead_linked"]
        batch_size = options["batch"]

        cutoff = timezone.now() - timezone.timedelta(days=days)
        qs = AIResult.objects.filter(created_at__lt=cutoff)

        if keep_linked:
            # Exclude AIResults pointed at by any still-open lead. We use the
            # `linked_leads` reverse relation defined on LeadRequest.ai_result.
            qs = qs.exclude(linked_leads__status__in=OPEN_LEAD_STATUSES).distinct()

        total = qs.count()
        if total == 0:
            self.stdout.write(self.style.SUCCESS(
                "Nothing to clean up — no AIResults older than "
                f"{days} day(s) matched."
            ))
            return

        self.stdout.write(
            f"Target: {total} AIResult row(s) older than {days} day(s) "
            f"(cutoff {cutoff.isoformat()})."
        )

        deleted_rows = 0
        deleted_files = 0
        freed_bytes = 0

        # Iterate over IDs in batches so we never hold a giant transaction.
        ids = list(qs.values_list("id", flat=True))
        for chunk_start in range(0, len(ids), batch_size):
            chunk_ids = ids[chunk_start : chunk_start + batch_size]

            # Collect the on-disk paths BEFORE deleting rows, so we still have
            # the ImageField values to resolve.
            paths = []
            for ai in AIResult.objects.filter(id__in=chunk_ids).only(
                "id", "image", "input_image"
            ):
                for field_val in (ai.image, ai.input_image):
                    if field_val and field_val.name:
                        try:
                            path = field_val.path
                        except Exception:
                            # FileField without filesystem storage — skip.
                            continue
                        paths.append(path)

            if dry_run:
                deleted_rows += len(chunk_ids)
                deleted_files += sum(1 for p in paths if os.path.exists(p))
                for p in paths:
                    try:
                        freed_bytes += os.path.getsize(p)
                    except OSError:
                        pass
                continue

            with transaction.atomic():
                AIResult.objects.filter(id__in=chunk_ids).delete()
                deleted_rows += len(chunk_ids)

            # Remove files AFTER the DB commit — if the delete rolled back,
            # we'd rather have orphan rows than orphan pointers to missing files.
            for path in paths:
                try:
                    size = os.path.getsize(path)
                    os.remove(path)
                    freed_bytes += size
                    deleted_files += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    self.stderr.write(f"  [skip file] {path}: {e}")

        freed_mb = freed_bytes / (1024 * 1024)
        label = "[DRY-RUN] Would delete" if dry_run else "Deleted"
        self.stdout.write(self.style.SUCCESS(
            f"{label} {deleted_rows} row(s), {deleted_files} file(s), "
            f"~{freed_mb:.1f} MB freed."
        ))
