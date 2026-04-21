import os
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from django.conf import settings
from shop.models import AIResult, AITest

class Command(BaseCommand):
    help = 'Cleans up local AI result and input images older than 24 hours, keeping them only on Telegram cloud.'

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(hours=24)
        
        # 1. Cleanup AIResult
        results = AIResult.objects.filter(created_at__lt=cutoff).exclude(telegram_file_id='')
        for res in results:
            updated = False
            if res.image:
                self.stdout.write(f"Deleting local result image for AIResult {res.id}")
                if os.path.exists(res.image.path):
                    os.remove(res.image.path)
                res.image = None
                updated = True
                
            if res.input_image:
                self.stdout.write(f"Deleting local input image for AIResult {res.id}")
                if os.path.exists(res.input_image.path):
                    os.remove(res.input_image.path)
                res.input_image = None
                updated = True
                
            if updated:
                res.save(update_fields=['image', 'input_image'])

        # 2. Cleanup AITest
        tests = AITest.objects.filter(created_at__lt=cutoff).exclude(telegram_file_id='')
        for test in tests:
            updated = False
            if test.result_image:
                self.stdout.write(f"Deleting local result image for AITest {test.id}")
                if os.path.exists(test.result_image.path):
                    os.remove(test.result_image.path)
                test.result_image = None
                updated = True
                
            if test.room_image:
                self.stdout.write(f"Deleting local room image for AITest {test.id}")
                if os.path.exists(test.room_image.path):
                    os.remove(test.room_image.path)
                test.room_image = None
                updated = True
                
            if updated:
                test.save(update_fields=['result_image', 'room_image'])

        self.stdout.write(self.style.SUCCESS('Successfully cleaned up old local AI files.'))
