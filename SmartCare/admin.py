from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils import timezone
from .models import Medication, Schedule, MedicationLog, RefillLog, PendingMedication, DrugInteraction, DrugInformation, FoodInteraction, IntakeLog


@admin.register(Medication)
class MedicationAdmin(admin.ModelAdmin):
    list_display = [
        'name', 
        'compartment_display_admin', 
        'compartment_number',  # ← ADDED for list_editable
        'dosage_count', 
        'pills_remaining', 
        'low_quantity_threshold',  # ← ADDED for list_editable
        'daily_consumption', 
        'days_supply', 
        'refill_status'
    ]
    list_filter = ['category', 'meal_instruction', 'compartment_number']
    search_fields = ['name', 'generic_name']
    list_editable = ['dosage_count', 'pills_remaining', 'low_quantity_threshold', 'compartment_number']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'generic_name', 'category', 'instructions')
        }),
        ('Dosage & Inventory', {
            'fields': ('dosage_count', 'total_pills_loaded', 'pills_remaining', 'low_quantity_threshold'),
            'description': 'For Arduino: dosage_count = number of pills per dose'
        }),
        ('Compartment Assignment', {
            'fields': ('compartment_number',),
            'description': '0 = Auto-Assign (based on order). Set 1-6 to assign to specific compartment.',
            'classes': ('wide',),
        }),
        ('Schedule Info', {
            'fields': ('daily_frequency_display', 'daily_consumption', 'days_supply'),
            'classes': ('collapse',),
            'description': 'Calculated from active schedules'
        }),
        ('Safety', {
            'fields': ('meal_instruction', 'interaction_notes')
        }),
    )
    
    readonly_fields = ['daily_frequency_display', 'daily_consumption', 'days_supply']
    
    def compartment_display_admin(self, obj):
        """Display compartment assignment with color coding"""
        if obj.compartment_number == 0:
            return mark_safe('<span style="color: #6b7280;">🔄 Auto-Assign</span>')
        return mark_safe(f'<span style="color: #2563eb; font-weight: bold;">📦 Comp {obj.compartment_number}</span>')
    compartment_display_admin.short_description = 'Compartment'
    compartment_display_admin.admin_order_field = 'compartment_number'
    
    def daily_frequency_display(self, obj):
        freq = obj.daily_frequency
        if freq == 0:
            return "No active schedules"
        return f"{freq}x daily"
    daily_frequency_display.short_description = 'Frequency'
    
    def daily_consumption(self, obj):
        consumption = obj.daily_pill_consumption
        if consumption == 0:
            return "N/A"
        return f"{consumption} pills/day"
    daily_consumption.short_description = 'Daily Consumption'
    
    def days_supply(self, obj):
        days = obj.days_of_supply
        if days <= 0:
            return mark_safe('<span style="color: red;">⚠️ No pills left</span>')
        elif days <= 3:
            return mark_safe(f'<span style="color: orange;">{days} days left</span>')
        else:
            return f"{days} days"
    days_supply.short_description = 'Days Supply'
    
    def refill_status(self, obj):
        if obj.pills_remaining <= obj.low_quantity_threshold:
            return mark_safe('<span style="color: red; font-weight: bold;">⚠️ Refill Needed</span>')
        return mark_safe('<span style="color: green;">✓ OK</span>')
    refill_status.short_description = 'Stock Status'
    refill_status.admin_order_field = 'pills_remaining'


@admin.register(Schedule)
class ScheduleAdmin(admin.ModelAdmin):
    list_display = [
        'medication', 
        'scheduled_time', 
        'compartment_display_admin',
        'dose_number_display',
        'start_date', 
        'supply_days_left', 
        'status',
        'intake_status_display',
        'last_verified_take'
    ]
    list_filter = ['status', 'intake_status', 'dose_number']
    search_fields = ['medication__name']
    list_editable = ['status']
    readonly_fields = ['supply_end_date', 'supply_days_left', 'daily_frequency_display']
    
    fieldsets = (
        ('Medication', {
            'fields': ('medication',)
        }),
        ('Schedule Time', {
            'fields': ('scheduled_time', 'dose_number', 'group_id')
        }),
        ('Date Range', {
            'fields': ('start_date',)
        }),
        ('Compartment Info', {
            'fields': ('compartment_display_admin',),
            'classes': ('collapse',),
        }),
        ('Supply Information', {
            'fields': ('daily_frequency_display', 'supply_end_date', 'supply_days_left'),
            'classes': ('collapse',),
        }),
        ('Intake Verification', {
            'fields': ('intake_status', 'last_verified_take', 'verification_attempts'),
            'classes': ('collapse',),
        }),
        ('Status', {
            'fields': ('status', 'notes', 'last_status_check')
        }),
    )
    
    def compartment_display_admin(self, obj):
        """Display compartment from medication"""
        comp = obj.medication.compartment_number
        if comp == 0:
            return mark_safe('<span style="color: #6b7280;">🔄 Auto</span>')
        return mark_safe(f'<span style="color: #2563eb;">📦 Comp {comp}</span>')
    compartment_display_admin.short_description = 'Compartment'
    
    def dose_number_display(self, obj):
        if obj.dose_number > 1:
            return f"Dose {obj.dose_number}"
        return "Once daily"
    dose_number_display.short_description = 'Dose'
    
    def daily_frequency_display(self, obj):
        freq = obj.daily_frequency
        if freq == 0:
            return "No active schedules"
        return f"{freq}x daily"
    daily_frequency_display.short_description = 'Frequency'
    
    def intake_status_display(self, obj):
        colors = {
            'pending': '#f59e0b',
            'verified': '#22c55e',
            'missed': '#ef4444',
            'late': '#f97316',
        }
        color = colors.get(obj.intake_status, '#64748b')
        return mark_safe(f'<span style="color: {color}; font-weight: bold;">{obj.get_intake_status_display()}</span>')
    intake_status_display.short_description = 'Intake Status'
    
    def supply_end_date(self, obj):
        end_date = obj.end_date
        if end_date:
            return end_date
        return 'Until pills run out'
    supply_end_date.short_description = 'Estimated End Date'
    
    def supply_days_left(self, obj):
        days = obj.days_remaining
        freq = obj.daily_frequency
        freq_text = f" ({freq}x daily)" if freq > 1 else ""
        if days <= 0:
            return mark_safe('<span style="color: red;">⚠️ No pills left</span>')
        elif days <= 3:
            return mark_safe(f'<span style="color: orange;">{days} days left{freq_text}</span>')
        else:
            return f"{days} days left{freq_text}"
    supply_days_left.short_description = 'Supply Remaining'


@admin.register(IntakeLog)
class IntakeLogAdmin(admin.ModelAdmin):
    list_display = [
        'medication', 
        'date', 
        'scheduled_time_display', 
        'actual_take_time_display',
        'status_display',
        'is_verified',
        'verification_method',
        'minutes_late'
    ]
    list_filter = ['status', 'is_verified', 'verification_method', 'date']
    search_fields = ['medication__name', 'schedule__medication__name']
    readonly_fields = ['created_at', 'updated_at']
    date_hierarchy = 'date'
    
    fieldsets = (
        ('Medication & Schedule', {
            'fields': ('schedule', 'medication')
        }),
        ('Timing', {
            'fields': ('scheduled_time', 'actual_take_time', 'date', 'minutes_late', 'time_to_take')
        }),
        ('Verification', {
            'fields': ('status', 'is_verified', 'verification_method', 'camera_confirmed', 'detection_confidence')
        }),
        ('Additional Info', {
            'fields': ('detection_metadata', 'notes', 'created_at', 'updated_at')
        }),
    )
    
    def scheduled_time_display(self, obj):
        if obj.scheduled_time:
            # Convert to local time and format
            local_time = timezone.localtime(obj.scheduled_time)
            return local_time.strftime('%I:%M %p')
        return '-'
    scheduled_time_display.short_description = 'Scheduled Time'
    
    def actual_take_time_display(self, obj):
        if obj.actual_take_time:
            local_time = timezone.localtime(obj.actual_take_time)
            return local_time.strftime('%I:%M %p')
        return '-'
    actual_take_time_display.short_description = 'Actual Take Time'
    
    def status_display(self, obj):
        colors = {
            'on_time': '#22c55e',
            'late': '#f97316',
            'missed': '#ef4444',
            'verified_camera': '#3b82f6',
            'verified_manual': '#8b5cf6',
            'failed_verification': '#ef4444',
        }
        color = colors.get(obj.status, '#64748b')
        return mark_safe(f'<span style="color: {color}; font-weight: bold;">{obj.get_status_display()}</span>')
    status_display.short_description = 'Status'
    
    def get_readonly_fields(self, request, obj=None):
        if obj:
            return self.readonly_fields + ['schedule', 'medication']
        return self.readonly_fields


@admin.register(MedicationLog)
class MedicationLogAdmin(admin.ModelAdmin):
    list_display = ['medication', 'dosage_taken', 'taken_at', 'verified_by', 'arduino_confirmed']
    list_filter = ['verified_by', 'arduino_confirmed']
    readonly_fields = ['taken_at']
    search_fields = ['medication__name']


@admin.register(RefillLog)
class RefillLogAdmin(admin.ModelAdmin):
    list_display = ['medication', 'quantity_added', 'new_quantity', 'refilled_at', 'refilled_by']
    list_filter = ['refilled_by']
    readonly_fields = ['refilled_at']
    search_fields = ['medication__name']
    

@admin.register(PendingMedication)
class PendingMedicationAdmin(admin.ModelAdmin):
    list_display = ['name', 'status', 'created_at', 'approval_token_short']
    list_filter = ['status', 'created_at']
    search_fields = ['name', 'approval_token']
    readonly_fields = ['approval_token', 'created_at', 'approved_at', 'interactions_found']
    
    def approval_token_short(self, obj):
        return obj.approval_token[:20] + '...' if obj.approval_token else '-'
    approval_token_short.short_description = 'Approval Token'
    
    fieldsets = (
        ('Medication Info', {
            'fields': ('name', 'generic_name', 'category', 'instructions')
        }),
        ('Dosage & Schedule', {
            'fields': ('dosage_count', 'total_pills_loaded', 'low_quantity_threshold', 
                      'meal_instruction', 'scheduled_time', 'multiple_times', 'start_date')
        }),
        ('Approval Info', {
            'fields': ('status', 'approval_token', 'interactions_found', 'created_at', 'approved_at')
        }),
    )


@admin.register(DrugInteraction)
class DrugInteractionAdmin(admin.ModelAdmin):
    list_display = ['drug_one', 'drug_two', 'description_short', 'description']  # ← ADDED description
    search_fields = ['drug_one', 'drug_two']
    list_filter = ['drug_one']
    list_editable = ['description']  # ← Now description is in list_display
    
    def description_short(self, obj):
        if len(obj.description) > 60:
            return obj.description[:60] + '...'
        return obj.description
    description_short.short_description = 'Description'


@admin.register(DrugInformation)
class DrugInformationAdmin(admin.ModelAdmin):
    list_display = ['name', 'manufacturer', 'excellent_review_percent', 'average_review_percent', 'poor_review_percent', 'medication']
    list_filter = ['manufacturer']
    search_fields = ['name', 'composition', 'uses', 'manufacturer']
    list_editable = ['excellent_review_percent', 'average_review_percent', 'poor_review_percent']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'composition', 'manufacturer', 'image_url', 'medication')
        }),
        ('Clinical Information', {
            'fields': ('uses', 'side_effects')
        }),
        ('Reviews', {
            'fields': ('excellent_review_percent', 'average_review_percent', 'poor_review_percent'),
            'description': 'Percentages should add up to 100%'
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at']


@admin.register(FoodInteraction)
class FoodInteractionAdmin(admin.ModelAdmin):
    list_display = ['drug_name', 'medication', 'interaction_type', 'severity', 'interaction_short']
    list_filter = ['interaction_type', 'severity']
    search_fields = ['drug_name', 'interaction_text']
    list_editable = ['interaction_type', 'severity']
    
    fieldsets = (
        ('Drug Information', {
            'fields': ('drug_name', 'medication')
        }),
        ('Interaction Details', {
            'fields': ('interaction_type', 'interaction_text', 'severity', 'reference')
        }),
    )
    
    def interaction_short(self, obj):
        if len(obj.interaction_text) > 60:
            return obj.interaction_text[:60] + '...'
        return obj.interaction_text
    interaction_short.short_description = 'Interaction'
    
    # admin.py
from django.contrib import admin
from .models import BehavioralData, BehaviorReport

@admin.register(BehavioralData)
class BehavioralDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'emotion', 'timestamp', 'adherence', 'behavior_score', 'is_synthetic', 'is_anomaly')
    list_filter = ('emotion', 'is_synthetic', 'is_anomaly')
    search_fields = ('emotion',)
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'

@admin.register(BehaviorReport)
class BehaviorReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'report_type', 'generated_at', 'dominant_emotion', 'average_adherence', 'anomaly_count', 'email_sent')
    list_filter = ('report_type', 'dominant_emotion', 'email_sent')
    search_fields = ('summary', 'recommendations')
    readonly_fields = ('generated_at',)
    date_hierarchy = 'generated_at'