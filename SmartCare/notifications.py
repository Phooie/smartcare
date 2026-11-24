import threading
import time
from datetime import datetime, timedelta
from django.utils import timezone
from django.db.models import Q
from .models import Schedule, IntakeLog
import logging

logger = logging.getLogger(__name__)

class NotificationService:
    """Service to manage medication notifications"""
    
    def __init__(self):
        self.notifications = {}
        self.running = False
        self.thread = None
    
    def check_scheduled_medications(self):
        """Check for medications that need notifications"""
        now = timezone.now()
        current_time = now.time()
        current_date = now.date()
        
        # Get active schedules for today
        active_schedules = Schedule.objects.filter(
            status='active',
            start_date__lte=current_date,
            medication__pills_remaining__gt=0,
        ).select_related('medication')
        
        upcoming_meds = []
        
        for schedule in active_schedules:
            # Check if schedule is active today
            if not schedule.is_active_today(current_date):
                continue
                
            # Check if already taken today
            today_logs = IntakeLog.objects.filter(
                schedule=schedule,
                date=current_date
            )
            
            if today_logs.exists():
                continue
            
            # Calculate time difference
            scheduled_datetime = datetime.combine(current_date, schedule.scheduled_time)
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
            
            time_diff = (now - scheduled_datetime).total_seconds() / 60  # in minutes
            
            # If within 1 hour before or after scheduled time
            if -60 <= time_diff <= 60:
                upcoming_meds.append({
                    'schedule': schedule,
                    'minutes_to_go': -time_diff,  # positive means late, negative means early
                    'status': 'on_time' if -15 <= time_diff <= 15 else 'late' if time_diff > 15 else 'early'
                })
        
        return upcoming_meds
    
    def create_medication_reminder(self, schedule_data):
        """Create a reminder notification for a medication"""
        schedule = schedule_data['schedule']
        minutes_to_go = schedule_data['minutes_to_go']
        
        return {
            'schedule_id': schedule.id,
            'medication_name': schedule.medication.name,
            'dosage': schedule.medication.dosage_count,
            'scheduled_time': schedule.scheduled_time.strftime('%I:%M %p'),
            'minutes_until': abs(int(minutes_to_go)),
            'is_late': minutes_to_go > 0,
            'status': schedule_data['status'],
            'created_at': timezone.now().isoformat()
        }
    
    def get_active_notifications(self):
        """Get all active notifications that should be shown"""
        upcoming = self.check_scheduled_medications()
        notifications = []
        
        for med_data in upcoming:
            reminder = self.create_medication_reminder(med_data)
            notifications.append(reminder)
        
        return notifications
    
    def mark_as_taken(self, schedule_id, method='manual', camera_confirmed=False):
        """Mark a medication as taken and create intake log"""
        try:
            schedule = Schedule.objects.get(id=schedule_id)
            now = timezone.now()
            scheduled_datetime = datetime.combine(now.date(), schedule.scheduled_time)
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
            
            minutes_late = int((now - scheduled_datetime).total_seconds() / 60)
            time_to_take = abs((now - scheduled_datetime).total_seconds())
            
            # Determine status
            if minutes_late <= 15:
                status = 'on_time'
            elif minutes_late <= 60:
                status = 'late'
            else:
                status = 'missed'
            
            # Update schedule
            schedule.last_verified_take = now
            schedule.verification_attempts += 1
            schedule.intake_status = 'verified' if camera_confirmed else 'pending'
            schedule.save()
            
            # Create intake log
            intake_log = IntakeLog.objects.create(
                schedule=schedule,
                medication=schedule.medication,
                scheduled_time=scheduled_datetime,
                actual_take_time=now,
                status=status,
                is_verified=camera_confirmed,
                verification_method=method,
                camera_confirmed=camera_confirmed,
                minutes_late=minutes_late if minutes_late > 0 else 0,
                time_to_take=int(time_to_take),
                notes=f"Verified via {method} at {now.strftime('%I:%M %p')}"
            )
            
            # Update medication inventory
            schedule.medication.dispense()
            
            return True, intake_log
            
        except Exception as e:
            logger.error(f"Error marking medication as taken: {e}")
            return False, None