from django.core.management.base import BaseCommand
from SmartCare.views import reset_daily_statuses

class Command(BaseCommand):
    help = 'Reset schedule statuses for the new day'

    def handle(self, *args, **options):
        reset_count = reset_daily_statuses()
        self.stdout.write(
            self.style.SUCCESS(f'✅ Reset {reset_count} schedule statuses for today')
        )