from django.db import models
from django.utils import timezone
from datetime import date, timedelta
from django.conf import settings
import secrets
from django.core.exceptions import ValidationError

class Medication(models.Model):
    MEAL_CHOICES = [
        ('before', 'Before Meal'),
        ('after', 'After Meal'),
        ('with', 'With Food'),
        ('empty', 'Empty Stomach'),
        ('any', 'No Restriction'),
    ]
    
    COMPARTMENT_CHOICES = [
        (0, 'Auto-Assign (based on order)'),
        (1, 'Compartment 1'),
        (2, 'Compartment 2'),
        (3, 'Compartment 3'),
        (4, 'Compartment 4'),
        (5, 'Compartment 5'),
        (6, 'Compartment 6'),
    ]
    
    compartment_number = models.IntegerField(
        default=0,
        choices=COMPARTMENT_CHOICES,
        help_text="0 = auto-assign based on order. Set 1-6 to assign manually."
    )
    
    name = models.CharField(max_length=200, help_text="e.g., Metformin 500mg")
    generic_name = models.CharField(max_length=200, blank=True)
    category = models.CharField(max_length=100, blank=True)
    
    # Dosage per intake (for Arduino motor rotation)
    dosage_count = models.IntegerField(default=1, help_text="Number of pills to dispense each time")
    
    # Total pills loaded
    total_pills_loaded = models.IntegerField(default=0, help_text="Total pills loaded into dispenser")
    pills_remaining = models.IntegerField(default=0, help_text="Pills still in dispenser")
    
    # Tracking
    low_quantity_threshold = models.IntegerField(default=5, help_text="Alert when pills drop below this")
    
    # Instructions
    meal_instruction = models.CharField(max_length=20, choices=MEAL_CHOICES, default='any')
    instructions = models.TextField(blank=True)
    
    # Interaction checking
    interaction_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        comp_display = "Auto" if self.compartment_number == 0 else f"Comp {self.compartment_number}"
        return f"{self.name} ({self.pills_remaining} pills left, {comp_display})"
    
    @property
    def compartment_display(self):
        """Get display name for compartment"""
        if self.compartment_number == 0:
            return "Auto-Assign"
        return f"Compartment {self.compartment_number}"
    
    @property
    def daily_frequency(self):
        """Get how many times per day this medication is taken"""
        return Schedule.objects.filter(
            medication=self,
            status='active'
        ).count()
    
    @property
    def daily_pill_consumption(self):
        """Get total pills consumed per day"""
        freq = self.daily_frequency
        if freq == 0:
            freq = 1
        return self.dosage_count * freq
    
    @property
    def days_of_supply(self):
        """Get number of days of supply remaining"""
        if self.daily_pill_consumption == 0:
            return 0
        return self.pills_remaining // self.daily_pill_consumption
    
    def calculate_end_date(self, start_date):
        """Calculate when pills will run out based on daily dosage and frequency"""
        if self.dosage_count == 0:
            return None
        
        schedules = Schedule.objects.filter(
            medication=self,
            status='active'
        )
        
        daily_frequency = schedules.count()
        if daily_frequency == 0:
            daily_frequency = 1
        
        daily_pill_consumption = self.dosage_count * daily_frequency
        
        if daily_pill_consumption == 0:
            return None
            
        days_of_supply = self.pills_remaining // daily_pill_consumption
        
        if days_of_supply > 0:
            return start_date + timedelta(days=days_of_supply - 1)
        return start_date
    
    def needs_refill(self):
        return self.pills_remaining <= self.low_quantity_threshold
    
    def dispense(self):
        """Decrease remaining pills when dispensed (pill is consumed)"""
        if self.pills_remaining >= self.dosage_count:
            self.pills_remaining -= self.dosage_count
            self.save()
            return True
        return False
    
    class Meta:
        ordering = ['name']


class Schedule(models.Model):
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('paused', 'Paused'),
    ]
    
    INTAKE_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('verified', 'Verified'),
        ('missed', 'Missed'),
        ('late', 'Late'),
    ]
    
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name='schedules')
    
    # Schedule timing
    scheduled_time = models.TimeField(help_text="Time to take the medication daily")
    start_date = models.DateField(default=date.today, help_text="Start taking from this date")
    
    # For multiple daily doses
    dose_number = models.IntegerField(default=1, help_text="Which dose of the day (1st, 2nd, 3rd, etc.)")
    group_id = models.CharField(max_length=50, blank=True, help_text="Group identifier for multiple daily doses")
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Intake verification fields
    last_verified_take = models.DateTimeField(null=True, blank=True)
    verification_attempts = models.IntegerField(default=0)
    intake_status = models.CharField(
        max_length=20,
        choices=INTAKE_STATUS_CHOICES,
        default='pending'
    )
    
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_status_check = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['scheduled_time']
        unique_together = ['medication', 'scheduled_time', 'start_date']
    
    def __str__(self):
        dose_info = f" (Dose {self.dose_number})" if self.dose_number > 1 else ""
        comp = self.medication.compartment_display
        return f"{self.medication.name}{dose_info} at {self.scheduled_time} → {comp}"
    
    @property
    def end_date(self):
        """Calculate when pills will run out"""
        return self.medication.calculate_end_date(self.start_date)
    
    @property
    def days_remaining(self):
        """Calculate how many days of pills left"""
        if self.medication.dosage_count == 0:
            return 0
        
        schedules = Schedule.objects.filter(
            medication=self.medication,
            status='active'
        )
        
        daily_frequency = schedules.count()
        if daily_frequency == 0:
            daily_frequency = 1
            
        daily_pill_consumption = self.medication.dosage_count * daily_frequency
        
        if daily_pill_consumption == 0:
            return 0
            
        return self.medication.pills_remaining // daily_pill_consumption
    
    @property
    def daily_frequency(self):
        """Get how many times per day this medication is taken"""
        return Schedule.objects.filter(
            medication=self.medication,
            status='active'
        ).count()
    
    def is_active_today(self, check_date=None):
        """Check if schedule should run today"""
        if check_date is None:
            check_date = date.today()
        
        if self.status != 'active':
            return False
        
        if check_date < self.start_date:
            return False
        
        if self.medication.pills_remaining <= 0:
            return False
        
        return True
    
    def mark_taken(self):
        """Mark as taken and update inventory"""
        from .models import MedicationLog
        if self.medication.dispense():
            self.intake_status = 'verified'
            self.last_verified_take = timezone.now()
            self.save()
            MedicationLog.objects.create(
                medication=self.medication,
                schedule=self,
                dosage_taken=self.medication.dosage_count,
                verified_by='user'
            )
            return True
        return False


class MedicationLog(models.Model):
    """Logs every pill intake"""
    schedule = models.ForeignKey(Schedule, on_delete=models.CASCADE, related_name='logs', null=True, blank=True)
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    dosage_taken = models.IntegerField(default=1)
    taken_at = models.DateTimeField(default=timezone.now)
    verified_by = models.CharField(max_length=50, default='user', help_text="user, camera, arduino, caregiver")
    
    arduino_confirmed = models.BooleanField(default=False)
    motor_rotation_count = models.IntegerField(default=0)
    
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.medication.name} taken at {self.taken_at}"
    
    class Meta:
        ordering = ['-taken_at']


class RefillLog(models.Model):
    """Track when pills are refilled"""
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE, related_name='refills')
    quantity_added = models.IntegerField()
    previous_quantity = models.IntegerField()
    new_quantity = models.IntegerField()
    refilled_at = models.DateTimeField(default=timezone.now)
    refilled_by = models.CharField(max_length=100, default='user')
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"Refilled {self.medication.name}: +{self.quantity_added}"
    
    class Meta:
        ordering = ['-refilled_at']


class DrugInteraction(models.Model):
    drug_one = models.CharField(max_length=255, db_index=True)
    drug_two = models.CharField(max_length=255, db_index=True)
    description = models.TextField()
    
    class Meta:
        unique_together = ['drug_one', 'drug_two']
        indexes = [
            models.Index(fields=['drug_one']),
            models.Index(fields=['drug_two']),
        ]
    
    def __str__(self):
        return f"{self.drug_one} + {self.drug_two}"
    

class PendingMedication(models.Model):
    """Stores medications waiting for caregiver approval"""
    STATUS_CHOICES = [
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('denied', 'Denied'),
    ]
    
    # Original medication data
    name = models.CharField(max_length=200)
    generic_name = models.CharField(max_length=200, blank=True)
    category = models.CharField(max_length=100, blank=True)
    instructions = models.TextField(blank=True)
    dosage_count = models.IntegerField(default=1)
    total_pills_loaded = models.IntegerField(default=0)
    low_quantity_threshold = models.IntegerField(default=5)
    meal_instruction = models.CharField(max_length=20, default='any')
    
    # Schedule data
    scheduled_time = models.TimeField(null=True, blank=True, help_text="Single time (for backward compatibility)")
    start_date = models.DateField()
    multiple_times = models.TextField(blank=True, help_text="Comma separated times for multiple daily doses")
    
    # Interaction info
    interactions_found = models.TextField(help_text="JSON of found interactions")
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    approval_token = models.CharField(max_length=64, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} - {self.status}"


class DrugInformation(models.Model):
    """Stores detailed drug information from datasets"""
    name = models.CharField(max_length=200, db_index=True, help_text="Medicine name")
    composition = models.TextField(blank=True, help_text="Active ingredients and strength")
    uses = models.TextField(blank=True, help_text="Therapeutic uses")
    side_effects = models.TextField(blank=True, help_text="Common side effects")
    image_url = models.URLField(max_length=500, blank=True, help_text="Product image URL")
    manufacturer = models.CharField(max_length=200, blank=True)
    excellent_review_percent = models.IntegerField(default=0, help_text="Excellent review %")
    average_review_percent = models.IntegerField(default=0, help_text="Average review %")
    poor_review_percent = models.IntegerField(default=0, help_text="Poor review %")
    
    medication = models.ForeignKey(
        'Medication', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='drug_information'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['manufacturer']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.manufacturer})"
    
    @property
    def total_review_percent(self):
        total = self.excellent_review_percent + self.average_review_percent + self.poor_review_percent
        return total


class FoodInteraction(models.Model):
    """Stores food interactions for medications"""
    INTERACTION_TYPE_CHOICES = [
        ('avoid', 'Avoid'),
        ('caution', 'Exercise Caution'),
        ('take_with', 'Take With'),
        ('take_without', 'Take Without'),
        ('take_separate', 'Take Separate'),
        ('supplement', 'Supplement Needed'),
        ('drink_fluids', 'Drink Fluids'),
        ('limit', 'Limit'),
        ('increase', 'Increase'),
        ('no_interaction', 'No Interaction'),
    ]
    
    medication = models.ForeignKey(
        'Medication', 
        on_delete=models.CASCADE, 
        related_name='food_interactions',
        null=True,
        blank=True
    )
    
    drug_name = models.CharField(max_length=200, db_index=True, help_text="Drug name")
    
    interaction_type = models.CharField(max_length=20, choices=INTERACTION_TYPE_CHOICES, default='avoid')
    interaction_text = models.TextField(help_text="Detailed interaction description")
    
    severity = models.CharField(max_length=20, choices=[
        ('mild', 'Mild'),
        ('moderate', 'Moderate'),
        ('severe', 'Severe'),
        ('unknown', 'Unknown'),
    ], default='unknown')
    
    reference = models.TextField(blank=True, help_text="Reference source")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['drug_name']
        indexes = [
            models.Index(fields=['drug_name']),
            models.Index(fields=['interaction_type']),
        ]
    
    def __str__(self):
        return f"{self.drug_name} - {self.interaction_text[:50]}..."
    

class IntakeLog(models.Model):
    """Detailed intake verification log"""
    INTAKE_STATUS_CHOICES = [
        ('on_time', 'On Time'),
        ('late', 'Late'),
        ('missed', 'Missed'),
        ('verified_camera', 'Verified by Camera'),
        ('verified_manual', 'Verified Manually'),
        ('failed_verification', 'Failed Verification'),
    ]
    
    schedule = models.ForeignKey(Schedule, on_delete=models.CASCADE, related_name='intake_logs')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    
    scheduled_time = models.DateTimeField()
    actual_take_time = models.DateTimeField(null=True, blank=True)
    date = models.DateField(default=date.today)  # ← CHANGED: Use date.today instead of auto_now_add
    
    status = models.CharField(max_length=30, choices=INTAKE_STATUS_CHOICES, default='missed')
    is_verified = models.BooleanField(default=False)
    verification_method = models.CharField(max_length=50, default='manual')
    
    camera_confirmed = models.BooleanField(default=False)
    detection_confidence = models.FloatField(default=0.0)
    detection_metadata = models.JSONField(default=dict, blank=True)
    
    minutes_late = models.IntegerField(default=0)
    time_to_take = models.IntegerField(help_text="Time in seconds from scheduled time to actual take", default=0)
    
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-date', '-scheduled_time']
        indexes = [
            models.Index(fields=['date', 'status']),
            models.Index(fields=['schedule', 'date']),
        ]
    
    def __str__(self):
        return f"{self.medication.name} - {self.date} - {self.status}"
    
    
class PendingMissedPill(models.Model):
    """Track missed pills awaiting caregiver decision"""
    STATUS_CHOICES = [
        ('pending', 'Pending Decision'),
        ('keep', 'Keep Pill (count stays)'),
        ('discard', 'Discard Pill (count reduced)'),
    ]
    
    schedule = models.ForeignKey(Schedule, on_delete=models.CASCADE, related_name='missed_pills')
    medication = models.ForeignKey(Medication, on_delete=models.CASCADE)
    scheduled_time = models.DateTimeField()
    missed_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    approval_token = models.CharField(max_length=64, unique=True, db_index=True, default=secrets.token_urlsafe)
    
    decision_made_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.medication.name} - {self.status} - {self.missed_at.date()}"
    
    
    # models.py - Add these models

class BehavioralData(models.Model):
    """Store behavioral data points for LSTM analysis"""
    
    EMOTION_CHOICES = [
        ('happy', 'Happy'),
        ('neutral', 'Neutral'),
        ('sad', 'Sad'),
        ('angry', 'Angry'),
        ('fear', 'Fear'),
        ('surprise', 'Surprise'),
        ('disgust', 'Disgust'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Emotion data
    emotion = models.CharField(max_length=20, choices=EMOTION_CHOICES, default='neutral')
    emotion_score = models.FloatField(default=0.5)  # 0=happy, 1=distressed
    confidence = models.FloatField(default=0.5)
    
    # Adherence data
    adherence = models.FloatField(default=0.5)  # 0-1
    missed_pills = models.IntegerField(default=0)
    stress_level = models.FloatField(default=0.5)  # 0-1
    
    # Combined behavior score
    behavior_score = models.FloatField(default=0.5)  # 0=good, 1=bad
    
    # Detection method
    is_synthetic = models.BooleanField(default=False)
    is_anomaly = models.BooleanField(default=False)
    anomaly_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['user', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.emotion} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class BehaviorReport(models.Model):
    """Store generated behavior reports"""
    
    REPORT_TYPES = [
        ('daily', 'Daily Report'),
        ('weekly', 'Weekly Report'),
        ('monthly', 'Monthly Report'),
        ('custom', 'Custom Report'),
    ]
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES, default='daily')
    generated_at = models.DateTimeField(auto_now_add=True)
    
    # Report data
    date_range_start = models.DateTimeField()
    date_range_end = models.DateTimeField()
    total_data_points = models.IntegerField(default=0)
    
    # Statistics
    average_emotion_score = models.FloatField(default=0.5)
    average_adherence = models.FloatField(default=0.5)
    total_missed_pills = models.IntegerField(default=0)
    anomaly_count = models.IntegerField(default=0)
    max_anomaly_score = models.FloatField(default=0.0)
    
    # Dominant emotions
    dominant_emotion = models.CharField(max_length=20, default='neutral')
    emotion_distribution = models.JSONField(default=dict)
    
    # Summary
    summary = models.TextField(blank=True)
    recommendations = models.TextField(blank=True)
    
    # Email tracking
    email_sent = models.BooleanField(default=False)
    email_sent_at = models.DateTimeField(null=True, blank=True)
    email_recipient = models.EmailField(blank=True)
    
    class Meta:
        ordering = ['-generated_at']
    
    def __str__(self):
        return f"{self.report_type} - {self.generated_at.strftime('%Y-%m-%d %H:%M')}"