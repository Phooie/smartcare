from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import date, timedelta
from SmartCare.models import Schedule, IntakeLog

class Command(BaseCommand):
    help = 'Auto-advance schedule when time has passed'

    def handle(self, *args, **options):
        today = date.today()
        now = timezone.now()
        
        # Get all active schedules
        schedules = Schedule.objects.filter(status='active')
        
        for schedule in schedules:
            # Check if schedule should have been taken today
            if schedule.start_date <= today and schedule.is_active_today(today):
                scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
                scheduled_datetime = timezone.make_aware(scheduled_datetime)
                
                # If schedule time has passed (more than 1 hour ago)
                time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
                
                if time_diff_minutes > 60:
                    # Check if it was taken
                    today_logs = IntakeLog.objects.filter(
                        schedule=schedule,
                        date=today
                    )
                    
                    if not today_logs.exists():
                        # Mark as missed
                        IntakeLog.objects.create(
                            schedule=schedule,
                            medication=schedule.medication,
                            scheduled_time=scheduled_datetime,
                            status='missed',
                            is_verified=False,
                            notes='Auto-marked as missed (time passed)'
                        )
                        schedule.intake_status = 'missed'
                        schedule.save()
                        self.stdout.write(f"Marked {schedule.medication.name} as MISSED")