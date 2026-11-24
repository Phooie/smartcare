from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, datetime, timedelta
from SmartCare.models import Schedule
from SmartCare.views import send_to_esp32

class Command(BaseCommand):
    help = 'Check for due medications and trigger alarm'

    def handle(self, *args, **options):
        today = date.today()
        now = timezone.now()
        
        # Get all active schedules for today
        schedules = Schedule.objects.filter(
            status='active',
            start_date__lte=today,
            medication__pills_remaining__gt=0,
            intake_status='pending'
        ).select_related('medication')
        
        for schedule in schedules:
            if not schedule.is_active_today(today):
                continue
            
            scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
            
            # Check if within 0-2 minutes of scheduled time
            time_diff = (now - scheduled_datetime).total_seconds() / 60
            
            if 0 <= time_diff <= 2:
                # Trigger alarm automatically
                self.stdout.write(f"🔔 Auto-triggering alarm for {schedule.medication.name}")
                
                # Send ALARM to ESP32
                success, response = send_to_esp32("ALARM")
                
                if success:
                    self.stdout.write(f"✅ Alarm triggered for {schedule.medication.name}")
                else:
                    self.stdout.write(f"❌ Failed to trigger alarm: {response}")