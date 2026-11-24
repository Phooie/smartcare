import json
import secrets
import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import time
import serial
from django.shortcuts import render, get_object_or_404, redirect
from django.utils import timezone
from django.contrib import messages
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.core.mail import send_mail
from django.conf import settings
from django.urls import reverse
from django.db import models
from datetime import date, datetime, timedelta
from .models import Medication, Schedule, MedicationLog, PendingMedication, FoodInteraction, DrugInteraction, IntakeLog
from .interaction_utils import check_drug_interactions

# ==================== GLOBAL VARIABLES ====================
hand_landmarker = None
face_landmarker = None
pill_detected_flag = False
esp32_serial = None
esp32_connected = False

# ==================== ESP32 SERIAL COMMUNICATION ====================

def get_esp32_connection():
    """Get serial connection to ESP32 on COM5"""
    global esp32_serial, esp32_connected
    
    if esp32_serial is not None:
        return esp32_connected, esp32_serial
    
    try:
        esp32_serial = serial.Serial('COM5', 115200, timeout=1)
        esp32_connected = True
        print("ESP32 connected on COM5")
        return esp32_connected, esp32_serial
        
    except Exception as e:
        print(f"Failed to connect on COM5: {e}")
        esp32_connected = False
        return False, None

def send_to_esp32(command):
    """Send command to ESP32 if connected"""
    connected, esp = get_esp32_connection()
    
    if not connected or esp is None:
        return False, "ESP32 not connected"
    
    try:
        esp.write(f"{command}\n".encode())
        time.sleep(0.3)
        
        response = ""
        if esp.in_waiting > 0:
            response = esp.readline().decode().strip()
        
        return True, response if response else "Command sent successfully"
    except Exception as e:
        return False, str(e)

# ==================== HAND-TO-MOUTH DETECTION ====================

def initialize_mediapipe():
    """Initialize MediaPipe models"""
    global hand_landmarker, face_landmarker
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        hand_model_path = os.path.join(base_dir, "hand_landmarker.task")
        face_model_path = os.path.join(base_dir, "face_landmarker.task")
        
        if not os.path.exists(hand_model_path):
            hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(hand_url, hand_model_path)
            print("Downloaded hand_landmarker.task")
        
        if not os.path.exists(face_model_path):
            face_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(face_url, face_model_path)
            print("Downloaded face_landmarker.task")
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        
        print("MediaPipe initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        return False


def detect_hand_to_mouth_with_landmarks(frame):
    """Detect hand-to-mouth gesture with landmarks and distance"""
    global hand_landmarker, face_landmarker
    
    if hand_landmarker is None or face_landmarker is None:
        return False, None, None, None, None
    
    try:
        INDEX_TIP = 8
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        hand_result = hand_landmarker.detect(mp_image)
        face_result = face_landmarker.detect(mp_image)
        
        mouth_center = None
        pill_detected = False
        hand_landmarks_list = None
        face_landmarks_list = None
        distance = None
        
        if face_result and face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            face_landmarks_list = []
            for lm in face_landmarks:
                face_landmarks_list.append({
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'z': lm.z
                })
            
            left_corner = face_landmarks[78]
            right_corner = face_landmarks[308]
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            
            mouth_center = (
                int(((left_corner.x + right_corner.x) / 2) * w),
                int(((upper_lip.y + lower_lip.y) / 2) * h)
            )
        
        if hand_result and hand_result.hand_landmarks and mouth_center:
            for hand_lms in hand_result.hand_landmarks:
                hand_landmarks_list = []
                for lm in hand_lms:
                    hand_landmarks_list.append({
                        'x': lm.x * w,
                        'y': lm.y * h,
                        'z': lm.z
                    })
                
                index_tip = hand_lms[INDEX_TIP]
                finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                distance = np.sqrt((finger_pos[0] - mouth_center[0])**2 + 
                                  (finger_pos[1] - mouth_center[1])**2)
                
                threshold = w * 0.08
                if distance < threshold:
                    pill_detected = True
                    break
        
        return pill_detected, mouth_center, hand_landmarks_list, face_landmarks_list, distance
        
    except Exception as e:
        print(f"Detection error: {e}")
        return False, None, None, None, None


def detect_hand_to_mouth_from_frame(frame):
    """Detect hand-to-mouth gesture from frame (simplified)"""
    pill_detected, _, _, _, _ = detect_hand_to_mouth_with_landmarks(frame)
    return pill_detected, None

# ==================== UPDATE STATUS FUNCTION ====================

def update_all_schedule_statuses():
    """Update statuses for all active schedules - resets daily and marks missed if time passed"""
    today = date.today()
    now = timezone.now()
    
    schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication')
    
    updated_count = 0
    missed_count = 0
    reset_count = 0
    
    for schedule in schedules:
        if not schedule.is_active_today(today):
            continue
        
        # Get today's log for this schedule
        today_logs = IntakeLog.objects.filter(
            schedule=schedule,
            date=today
        )
        
        # Calculate time difference
        scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
        scheduled_datetime = timezone.make_aware(scheduled_datetime)
        time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
        
        # ===== RESET: If time is in the FUTURE, mark as pending =====
        if time_diff_minutes < 0:
            if schedule.intake_status != 'pending':
                print(f"Resetting {schedule.medication.name} at {schedule.scheduled_time} to pending (future time)")
                schedule.intake_status = 'pending'
                schedule.save()
                reset_count += 1
            # Skip further processing for future times
            continue
        
        # ===== KEEP VERIFIED: If already verified and has a log =====
        if schedule.intake_status == 'verified' and today_logs.exists():
            continue
        
        # ===== CREATE LOG: If verified but no log and time has passed =====
        if schedule.intake_status == 'verified' and not today_logs.exists():
            IntakeLog.objects.create(
                schedule=schedule,
                medication=schedule.medication,
                scheduled_time=scheduled_datetime,
                actual_take_time=schedule.last_verified_take or timezone.now(),
                date=today,
                status='on_time',
                is_verified=True,
                verification_method='manual',
                notes='Auto-created log from verification status'
            )
            continue
        
        # Skip if already has a log for today
        if today_logs.exists():
            continue
        
        # ===== MARK MISSED: If more than 60 minutes past =====
        if time_diff_minutes > 60:
            print(f"Marking {schedule.medication.name} at {schedule.scheduled_time} as MISSED ({time_diff_minutes:.1f} min late)")
            schedule.intake_status = 'missed'
            schedule.save()
            
            IntakeLog.objects.create(
                schedule=schedule,
                medication=schedule.medication,
                scheduled_time=scheduled_datetime,
                date=today,
                status='missed',
                is_verified=False,
                minutes_late=int(time_diff_minutes),
                notes=f'Auto-marked as missed. Pill not consumed - remains in dispenser.'
            )
            updated_count += 1
            missed_count += 1
    
    return {
        'updated_count': updated_count,
        'missed_count': missed_count,
        'reset_count': reset_count,
    }

# ==================== VIEWS ====================

def main(request):
    """Dashboard with real data"""
    today = date.today()
    update_all_schedule_statuses()
    
    all_schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication')
    
    today_schedules = [s for s in all_schedules if s.is_active_today(today)]
    
    total_today = len(today_schedules)
    taken_today = len([s for s in today_schedules if s.intake_status == 'verified'])
    compliance = int((taken_today / total_today) * 100) if total_today > 0 else 0
    
    drug_names = list(set([s.medication.name.lower().strip() for s in today_schedules]))
    drug_interactions = check_drug_interactions(drug_names)
    
    food_interactions = []
    for med_name in drug_names:
        food_ints = FoodInteraction.objects.filter(
            drug_name__icontains=med_name
        )
        for fi in food_ints:
            food_interactions.append({
                'medication': med_name,
                'interaction_text': fi.interaction_text,
                'interaction_type': fi.get_interaction_type_display(),
                'severity': fi.get_severity_display(),
                'drug_name': fi.drug_name
            })
    
    low_stock_meds = Medication.objects.filter(
        pills_remaining__lte=models.F('low_quantity_threshold'),
        pills_remaining__gt=0
    )
    out_of_stock_meds = Medication.objects.filter(pills_remaining=0)
    
    context = {
        'today_total': total_today,
        'today_taken': taken_today,
        'compliance': compliance,
        'drug_interactions': drug_interactions,
        'food_interactions': food_interactions,
        'low_stock_count': low_stock_meds.count(),
        'out_of_stock_count': out_of_stock_meds.count(),
        'today_schedules': today_schedules[:5],
        'has_drug_interactions': len(drug_interactions) > 0,
        'has_food_interactions': len(food_interactions) > 0,
    }
    return render(request, 'main.html', context)


def schedule(request):
    """Display today's medication schedule with separate interaction sections"""
    update_all_schedule_statuses()  # ← THIS MUST BE HERE    
    today = date.today()
    
    all_schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication').order_by('scheduled_time')
    
    today_schedules = []
    for s in all_schedules:
        if s.is_active_today(today):
            today_schedules.append(s)
    
    if not today_schedules:
        context = {
            'schedules': [],
            'compliance': 0,
            'total_pills': 0,
            'taken_pills': 0,
            'today': today,
            'drug_interactions': [],
            'food_interactions': [],
            'has_drug_interactions': False,
            'has_food_interactions': False,
            'next_medications': [],
            'pre_notifications': [],
            'graph_labels_json': json.dumps([]),
            'medication_datasets_json': json.dumps([]),
            'has_graph_data': False,
        }
        return render(request, 'schedule.html', context)
    
    # ====== GENERATE GRAPH DATA ======
    graph_labels = []
    medication_datasets = []
    
    medication_ids = list(set([s.medication.id for s in today_schedules]))
    
    if medication_ids:
        for i in range(13, -1, -1):
            check_date = today - timedelta(days=i)
            graph_labels.append(check_date.strftime('%a %d'))
        
        colors = ['#2563eb', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
        
        for idx, med_id in enumerate(medication_ids):
            try:
                med = Medication.objects.get(id=med_id)
                data_points = []
                
                for i in range(13, -1, -1):
                    check_date = today - timedelta(days=i)
                    count = IntakeLog.objects.filter(
                        medication=med,
                        date=check_date,
                        status__in=['on_time', 'late', 'verified_camera', 'verified_manual']
                    ).count()
                    data_points.append(count)
                
                medication_datasets.append({
                    'label': med.name,
                    'data': data_points,
                    'borderColor': colors[idx % len(colors)],
                    'backgroundColor': colors[idx % len(colors)] + '33',
                    'borderWidth': 2,
                    'pointRadius': 3,
                    'pointBackgroundColor': colors[idx % len(colors)],
                    'tension': 0.3,
                    'fill': False,
                })
            except Medication.DoesNotExist:
                pass
    
    drug_names = list(set([s.medication.name.lower().strip() for s in today_schedules]))
    drug_interactions = check_drug_interactions(drug_names)
    
    food_interactions = []
    seen_medications = set()

    for med in today_schedules:
        med_name = med.medication.name.lower().strip()
        
        if med_name in seen_medications:
            continue
        seen_medications.add(med_name)
        
        food_ints = FoodInteraction.objects.filter(
            drug_name__icontains=med_name
        )
        for fi in food_ints:
            food_interactions.append({
                'medication': med.medication.name,
                'food_item': fi.drug_name,
                'interaction_text': fi.interaction_text,
                'interaction_type': fi.get_interaction_type_display(),
                'severity': fi.get_severity_display(),
            })
    
    total = len(today_schedules)
    taken = len([s for s in today_schedules if s.intake_status == 'verified'])
    compliance = int((taken / total) * 100) if total > 0 else 0
    
    # ====== MULTIPLE NOTIFICATIONS ======
    next_medications = []
    pre_notifications = []
    now = timezone.now()
    
    
    
    for schedule in today_schedules:
        if schedule.intake_status in ['verified', 'missed']:
            continue
            
        scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
        scheduled_datetime = timezone.make_aware(scheduled_datetime)
        time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
        
        # Pre-notification (5 minutes before)
        if -5 <= time_diff_minutes < 0:
            time_until_seconds = abs(time_diff_minutes) * 60
            hours = int(time_until_seconds // 3600)
            minutes = int((time_until_seconds % 3600) // 60)
            seconds = int(time_until_seconds % 60)
            pre_countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            pre_notifications.append({
                'schedule_id': schedule.id,
                'medication_name': schedule.medication.name,
                'dosage': schedule.medication.dosage_count,
                'scheduled_time': schedule.scheduled_time.strftime('%I:%M %p'),
                'target_datetime': scheduled_datetime.isoformat(),
                'countdown': pre_countdown,
            })
        
        # Main notification (0 to 60 minutes after)
        if 0 <= time_diff_minutes <= 60:
            is_late = time_diff_minutes > 15
            time_remaining_seconds = (60 - time_diff_minutes) * 60
            hours = int(time_remaining_seconds // 3600)
            minutes = int((time_remaining_seconds % 3600) // 60)
            seconds = int(time_remaining_seconds % 60)
            countdown = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            next_medications.append({
                'schedule_id': schedule.id,
                'medication_name': schedule.medication.name,
                'dosage': schedule.medication.dosage_count,
                'scheduled_time': schedule.scheduled_time.strftime('%I:%M %p'),
                'target_datetime': scheduled_datetime.isoformat(),
                'countdown': countdown,
                'is_late': is_late,
                'time_elapsed_minutes': int(time_diff_minutes),
                'time_remaining_minutes': int(60 - time_diff_minutes),
            })
    
    context = {
        'schedules': today_schedules,
        'compliance': compliance,
        'total_pills': total,
        'taken_pills': taken,
        'today': today,
        'drug_interactions': drug_interactions,
        'food_interactions': food_interactions,
        'has_drug_interactions': len(drug_interactions) > 0,
        'has_food_interactions': len(food_interactions) > 0,
        'next_medications': next_medications,
        'pre_notifications': pre_notifications,
        'graph_labels_json': json.dumps(graph_labels),
        'medication_datasets_json': json.dumps(medication_datasets),
        'has_graph_data': len(medication_datasets) > 0,
    }
    return render(request, 'schedule.html', context)

def intake_verification(request):
    """Intake verification page with camera and detection"""
    today = date.today()
    now = timezone.now()
    
    update_all_schedule_statuses()  # ← KEEP ONLY THIS ONE
    
    # REMOVED THE DUPLICATE: update_all_schedule_statuses()  # ← DELETE THIS LINE
    
    schedules_to_check = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication')
    
    for schedule in schedules_to_check:
        if not schedule.is_active_today(today):
            continue
            
        scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
        scheduled_datetime = timezone.make_aware(scheduled_datetime)
        time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
        
        if time_diff_minutes > 60:
            today_logs = IntakeLog.objects.filter(
                schedule=schedule,
                date=today
            )
            
            if not today_logs.exists() and schedule.intake_status != 'verified':
                if schedule.intake_status != 'missed':
                    IntakeLog.objects.create(
                        schedule=schedule,
                        medication=schedule.medication,
                        scheduled_time=scheduled_datetime,
                        date=today,
                        status='missed',
                        is_verified=False,
                        minutes_late=int(time_diff_minutes),
                        notes=f'Auto-marked as missed. Pill not consumed - remains in dispenser.'
                    )
                    schedule.intake_status = 'missed'
                    schedule.save()
    
    current_schedule = None
    current_schedule_id = None
    no_medication_today = True
    next_schedule_display = None
    is_verification_time = False
    
    active_schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication').order_by('scheduled_time')
    
    today_schedules = [s for s in active_schedules if s.is_active_today(today)]
    
    if today_schedules:
        no_medication_today = False
        
        for schedule in today_schedules:
            scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
            time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
            
            is_in_window = -5 <= time_diff_minutes <= 60
            
            today_logs = IntakeLog.objects.filter(
                schedule=schedule,
                date=today
            )
            
            if is_in_window and not today_logs.exists() and schedule.intake_status != 'verified':
                current_schedule = schedule
                current_schedule_id = schedule.id
                next_schedule_display = schedule.scheduled_time.strftime('%I:%M %p')
                if schedule.dose_number > 1:
                    next_schedule_display += f" (Dose {schedule.dose_number})"
                is_verification_time = True
                break
        
        if not current_schedule:
            for schedule in today_schedules:
                scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
                scheduled_datetime = timezone.make_aware(scheduled_datetime)
                time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
                
                if time_diff_minutes < -5:
                    next_schedule_display = schedule.scheduled_time.strftime('%I:%M %p')
                    if schedule.dose_number > 1:
                        next_schedule_display += f" (Dose {schedule.dose_number})"
                    break
            
            if not next_schedule_display:
                tomorrow = today + timedelta(days=1)
                tomorrow_schedules = Schedule.objects.filter(
                    status='active',
                    start_date__lte=tomorrow,
                    medication__pills_remaining__gt=0
                ).order_by('scheduled_time')
                for schedule in tomorrow_schedules:
                    if schedule.is_active_today(tomorrow):
                        next_schedule_display = schedule.scheduled_time.strftime('%I:%M %p') + ' (Tomorrow)'
                        if schedule.dose_number > 1:
                            next_schedule_display += f" (Dose {schedule.dose_number})"
                        break
    
    recent_logs = IntakeLog.objects.filter(
        date__gte=today - timedelta(days=7)
    ).select_related('medication').order_by('-date')[:10]
    
    graph_labels = []
    on_time_data = []
    late_data = []
    missed_data = []
    
    for i in range(6, -1, -1):
        check_date = today - timedelta(days=i)
        graph_labels.append(check_date.strftime('%a %d'))
        
        day_logs = IntakeLog.objects.filter(date=check_date)
        on_time_count = day_logs.filter(status='on_time').count()
        late_count = day_logs.filter(status='late').count()
        missed_count = day_logs.filter(status='missed').count()
        
        on_time_data.append(on_time_count)
        late_data.append(late_count)
        missed_data.append(missed_count)
    
    context = {
        'current_medication': current_schedule,
        'current_schedule_id': current_schedule_id,
        'recent_logs': recent_logs,
        'graph_labels_json': json.dumps(graph_labels),
        'on_time_data_json': json.dumps(on_time_data),
        'late_data_json': json.dumps(late_data),
        'missed_data_json': json.dumps(missed_data),
        'is_verification_time': is_verification_time,
        'no_medication_today': no_medication_today,
        'next_schedule_display': next_schedule_display,
    }
    
    return render(request, 'intake_verification.html', context)


# ==================== ESP32 CONTROL VIEWS ====================

@csrf_exempt
def trigger_medication_dose(request, schedule_id):
    """Trigger dispensation AFTER user verifies"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    schedule = get_object_or_404(Schedule, id=schedule_id)
    today = date.today()
    
    now = timezone.now()
    scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
    scheduled_datetime = timezone.make_aware(scheduled_datetime)
    
    time_diff = (now - scheduled_datetime).total_seconds() / 60
    
    if time_diff < -5 or time_diff > 60:
        return JsonResponse({
            'status': 'error', 
            'message': f'Not within medication window (5 min before to 60 min after)'
        })
    
    compartment = getattr(schedule.medication, 'compartment_number', 1)
    if compartment == 0:
        compartment = 1
    
    print(f"Triggering dose for {schedule.medication.name} - Compartment {compartment}")
    
    # Send OPEN command to ESP32
    success, response = send_to_esp32(f"OPEN {compartment}")
    
    if not success:
        return JsonResponse({
            'status': 'error',
            'message': f'ESP32 error: {response}'
        })
    
    return JsonResponse({
        'status': 'success',
        'message': f'Please take {schedule.medication.name} from compartment {compartment}',
        'compartment': compartment,
        'medication_name': schedule.medication.name
    })

@csrf_exempt
def close_compartments(request):
    """Close all ESP32 compartments"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    success, response = send_to_esp32("CLOSE")
    
    if success:
        return JsonResponse({
            'status': 'success',
            'message': 'All compartments closed'
        })
    else:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to close: {response}'
        })

@csrf_exempt
def test_esp32(request):
    """Test ESP32 connection"""
    connected, esp = get_esp32_connection()
    
    if not connected:
        return JsonResponse({
            'status': 'error',
            'message': 'ESP32 not connected. Check USB cable and COM port.'
        })
    
    success, response = send_to_esp32("STATUS")
    
    return JsonResponse({
        'status': 'success' if success else 'error',
        'esp32_response': response,
        'message': 'ESP32 is connected and responding!' if success else 'ESP32 not responding'
    })


# ==================== DETECTION VIEWS ====================

@csrf_exempt
def detect_gesture(request):
    """Detect hand-to-mouth gesture from a single frame"""
    global pill_detected_flag
    
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        data = json.loads(request.body)
        image_data = data.get('image')
        schedule_id = data.get('schedule_id')
        
        print(f"📸 Detection request - schedule_id: {schedule_id}")
        
        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'Missing image'}, status=400)
        
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.replace('data:image/jpeg;base64,', '')
        elif image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Failed to decode image: {str(e)}'}, status=400)
        
        if frame is None:
            return JsonResponse({'status': 'error', 'message': 'Invalid image data'}, status=400)
        
        if not initialize_mediapipe():
            return JsonResponse({'status': 'error', 'message': 'Failed to initialize detection'}, status=500)
        
        # Get detection results with landmarks
        pill_detected, mouth_center, hand_landmarks, face_landmarks, distance = detect_hand_to_mouth_with_landmarks(frame)
        
        hand_detected = hand_landmarks is not None
        mouth_detected = face_landmarks is not None
        gesture_detected = pill_detected
        
        print(f"🔍 Detection: hand={hand_detected}, mouth={mouth_detected}, gesture={gesture_detected}, distance={distance}")
        
        # If pill detected, set the flag and close ESP32
        if pill_detected:
            pill_detected_flag = True
            success, response = send_to_esp32("CLOSE")
            if success:
                print("✅ ESP32 closed via gesture detection")
            else:
                print(f"⚠️ Failed to close ESP32: {response}")
        
        # If pill detected and we have a schedule ID, verify
        if pill_detected and schedule_id:
            try:
                schedule = Schedule.objects.get(id=schedule_id)
                print(f"✅ Found schedule: {schedule.medication.name}")
                
                if schedule.intake_status == 'verified':
                    print(f"⚠️ Already verified: {schedule.medication.name}")
                    return JsonResponse({
                        'status': 'success',
                        'hand_detected': hand_detected,
                        'mouth_detected': mouth_detected,
                        'gesture_detected': gesture_detected,
                        'hand_landmarks': hand_landmarks,
                        'face_landmarks': face_landmarks,
                        'distance': distance,
                        'verified': False,
                        'message': 'Already verified'
                    })
                
                if schedule.medication.pills_remaining < schedule.medication.dosage_count:
                    print(f"❌ Not enough pills. Remaining: {schedule.medication.pills_remaining}")
                    return JsonResponse({
                        'status': 'success',
                        'hand_detected': hand_detected,
                        'mouth_detected': mouth_detected,
                        'gesture_detected': gesture_detected,
                        'hand_landmarks': hand_landmarks,
                        'face_landmarks': face_landmarks,
                        'distance': distance,
                        'verified': False,
                        'message': 'Not enough pills remaining'
                    })
                
                now = timezone.now()
                today = date.today()
                scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
                scheduled_datetime = timezone.make_aware(scheduled_datetime)
                time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
                
                if time_diff_minutes > 60:
                    print(f"⚠️ Too late! {time_diff_minutes} minutes past.")
                    return JsonResponse({
                        'status': 'success',
                        'hand_detected': hand_detected,
                        'mouth_detected': mouth_detected,
                        'gesture_detected': gesture_detected,
                        'hand_landmarks': hand_landmarks,
                        'face_landmarks': face_landmarks,
                        'distance': distance,
                        'verified': False,
                        'message': 'Medication window expired'
                    })
                
                is_late = time_diff_minutes > 15 and time_diff_minutes <= 60
                status = 'late' if is_late else 'on_time'
                
                schedule.medication.dispense()
                schedule.intake_status = 'verified'
                schedule.last_verified_take = now
                schedule.save()
                
                print(f"✅ Pill dispensed. New remaining: {schedule.medication.pills_remaining}")
                
                IntakeLog.objects.create(
                    schedule=schedule,
                    medication=schedule.medication,
                    scheduled_time=scheduled_datetime,
                    actual_take_time=now,
                    date=today,  # ← ADD THIS
                    status=status,
                    is_verified=True,
                    verification_method='camera',
                    camera_confirmed=True,
                    detection_confidence=0.95,
                    minutes_late=int(time_diff_minutes) if time_diff_minutes > 0 else 0,
                    notes=f'Verified via hand-to-mouth detection. {"Late" if is_late else "On time"}'
                )
                
                print(f"✅ Intake log created with status: {status}")
                
                # Send CLOSE command again to be safe
                send_to_esp32("CLOSE")
                
                return JsonResponse({
                    'status': 'success',
                    'hand_detected': hand_detected,
                    'mouth_detected': mouth_detected,
                    'gesture_detected': gesture_detected,
                    'hand_landmarks': hand_landmarks,
                    'face_landmarks': face_landmarks,
                    'distance': distance,
                    'verified': True,
                    'is_late': is_late,
                    'message': f'✅ Pill intake verified for {schedule.medication.name} {"(Late)" if is_late else ""}',
                    'pills_remaining': schedule.medication.pills_remaining
                })
                
            except Schedule.DoesNotExist:
                print(f"❌ Schedule not found with ID: {schedule_id}")
                return JsonResponse({
                    'status': 'success',
                    'hand_detected': hand_detected,
                    'mouth_detected': mouth_detected,
                    'gesture_detected': gesture_detected,
                    'hand_landmarks': hand_landmarks,
                    'face_landmarks': face_landmarks,
                    'distance': distance,
                    'verified': False,
                    'message': 'Schedule not found'
                })
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return JsonResponse({
                    'status': 'success',
                    'hand_detected': hand_detected,
                    'mouth_detected': mouth_detected,
                    'gesture_detected': gesture_detected,
                    'hand_landmarks': hand_landmarks,
                    'face_landmarks': face_landmarks,
                    'distance': distance,
                    'verified': False,
                    'message': str(e)
                })
        
        return JsonResponse({
            'status': 'success',
            'hand_detected': hand_detected,
            'mouth_detected': mouth_detected,
            'gesture_detected': gesture_detected,
            'hand_landmarks': hand_landmarks,
            'face_landmarks': face_landmarks,
            'distance': distance,
            'verified': False
        })
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'}, status=400)
    except Exception as e:
        print(f"❌ Error in detect_gesture: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def get_detection_status(request):
    """Get detection status and auto-close ESP32 compartments when verified"""
    global pill_detected_flag
    
    if pill_detected_flag:
        pill_detected_flag = False
        schedule_id = request.GET.get('schedule_id')
        
        if schedule_id:
            try:
                schedule = Schedule.objects.get(id=schedule_id)
                
                if schedule.intake_status != 'verified':
                    if schedule.medication.pills_remaining < schedule.medication.dosage_count:
                        return JsonResponse({
                            'status': 'error',
                            'pill_detected': True,
                            'message': 'Not enough pills remaining! Please refill.'
                        })
                    
                    # Mark as taken
                    schedule.medication.dispense()
                    schedule.intake_status = 'verified'
                    schedule.last_verified_take = timezone.now()
                    schedule.save()
                    
                    # Close ESP32 compartments
                    success, response = send_to_esp32("CLOSE")
                    
                    if success:
                        print("✅ ESP32 compartments closed")
                    else:
                        print(f"⚠️ Failed to close ESP32: {response}")
                    
                    time.sleep(0.3)
                    send_to_esp32("CLOSE")
                    
                    return JsonResponse({
                        'status': 'success',
                        'pill_detected': True,
                        'verified': True,
                        'message': f'✅ Pill taken! Compartments closed.',
                        'servo_action': 'closed',
                        'esp32_response': response,
                        'pills_remaining': schedule.medication.pills_remaining
                    })
                else:
                    return JsonResponse({
                        'status': 'success',
                        'pill_detected': True,
                        'verified': False,
                        'message': 'This medication was already verified.',
                        'servo_action': 'already_verified'
                    })
                    
            except Schedule.DoesNotExist:
                return JsonResponse({
                    'status': 'error',
                    'pill_detected': True,
                    'message': 'Schedule not found'
                }, status=404)
            except Exception as e:
                print(f"Error in get_detection_status: {e}")
                return JsonResponse({
                    'status': 'error',
                    'pill_detected': True,
                    'message': str(e)
                }, status=500)
    
    return JsonResponse({
        'status': 'success',
        'pill_detected': False,
        'message': 'Waiting for pill intake...'
    })


@csrf_exempt
def verify_intake(request):
    """Verify pill intake manually"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)
    
    try:
        data = json.loads(request.body)
        schedule_id = data.get('schedule_id')
        
        if not schedule_id:
            return JsonResponse({'status': 'error', 'message': 'Missing schedule ID'}, status=400)
        
        schedule = get_object_or_404(Schedule, id=schedule_id)
        
        if schedule.medication.pills_remaining < schedule.medication.dosage_count:
            return JsonResponse({
                'status': 'error',
                'message': 'Not enough pills remaining. Please refill.'
            }, status=400)
        
        now = timezone.now()
        scheduled_datetime = datetime.combine(date.today(), schedule.scheduled_time)
        scheduled_datetime = timezone.make_aware(scheduled_datetime)
        time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
        
        is_late = time_diff_minutes > 15
        status = 'late' if is_late else 'on_time'
        
        schedule.medication.dispense()
        schedule.intake_status = 'verified'
        schedule.last_verified_take = now
        schedule.save()
        
        IntakeLog.objects.create(
            schedule=schedule,
            medication=schedule.medication,
            scheduled_time=scheduled_datetime,
            actual_take_time=now,
            date=date.today(),  # ← ADD THIS
            status=status,
            is_verified=True,
            verification_method='manual',
            minutes_late=int(time_diff_minutes) if time_diff_minutes > 0 else 0,
            notes=f'Verified manually. {"Late" if is_late else "On time"}'
        )
        
        # Close ESP32 compartments
        send_to_esp32("CLOSE")
        
        return JsonResponse({
            'status': 'success',
            'is_late': is_late,
            'message': f'Pill intake verified for {schedule.medication.name} {"(Late)" if is_late else ""}'
        })
        
    except Schedule.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Schedule not found'}, status=404)
    except Exception as e:
        print(f"Error in verify_intake: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


# ==================== REMOVAL FUNCTIONS ====================

@csrf_protect
def request_remove_medication(request, schedule_id):
    """Request caregiver approval to remove a medication"""
    schedule = get_object_or_404(Schedule, id=schedule_id)
    
    if request.method == 'POST':
        approval_token = secrets.token_urlsafe(32)
        
        request.session['pending_removal_id'] = schedule.id
        request.session['pending_removal_token'] = approval_token
        
        approval_url = request.build_absolute_uri(
            reverse('approve_removal', args=[approval_token, schedule_id])
        )
        denial_url = request.build_absolute_uri(
            reverse('deny_removal', args=[approval_token, schedule_id])
        )
        
        if getattr(settings, 'CAREGIVER_EMAIL', None):
            try:
                send_mail(
                    subject=f'[SmartCare] Removal Request: {schedule.medication.name}',
                    message=f"""
A request has been made to remove {schedule.medication.name} from the schedule.

Please review and approve or deny this request:

✅ Approve Removal: {approval_url}
❌ Deny Removal: {denial_url}
                    """,
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[settings.CAREGIVER_EMAIL],
                    fail_silently=True,
                )
            except Exception as e:
                print(f"Email error: {e}")
        
        messages.warning(request, 
            f"⚠️ Removal request for {schedule.medication.name} has been sent to your caregiver.")
        return redirect('schedule')
    
    return redirect('schedule')


def approve_removal(request, token, schedule_id):
    """Caregiver approval for removal"""
    schedule = get_object_or_404(Schedule, id=schedule_id)
    
    if request.method == 'POST':
        schedule.status = 'completed'
        schedule.save()
        messages.success(request, f"✅ {schedule.medication.name} has been removed from schedule.")
        return redirect('schedule')
    
    context = {
        'schedule': schedule,
        'token': token,
    }
    return render(request, 'approve_removal.html', context)


def deny_removal(request, token, schedule_id):
    """Caregiver denial for removal"""
    schedule = get_object_or_404(Schedule, id=schedule_id)
    messages.info(request, f"❌ Removal of {schedule.medication.name} has been denied by caregiver.")
    return redirect('schedule')


# ==================== MEDICATION FUNCTIONS ====================

# @csrf_protect
@csrf_exempt
def add_medication(request):
    """Step 1: Check interactions before saving - with duplicate check"""
    if request.method != 'POST':
        return redirect('schedule')
    
    today = date.today()
    new_medication_name = request.POST.get('name', '').lower().strip()
    
    # Get existing active medications
    existing_schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication')
    
    existing_drugs = list(set([s.medication.name.lower().strip() for s in existing_schedules]))
    
    # ===== CHECK ONLY NEW MEDICATION AGAINST EXISTING =====
    interactions = []
    for existing_drug in existing_drugs:
        if existing_drug == new_medication_name:
            continue
        
        # Check for interactions using the interaction_utils
        result = check_drug_interactions([existing_drug, new_medication_name])
        if result:
            for r in result:
                # ===== ONLY CHECK FOR PENDING APPROVAL =====
                # Don't check if it exists in DrugInteraction - that's where it comes from!
                pending_exists = PendingMedication.objects.filter(
                    status='pending',
                    name__icontains=new_medication_name
                ).exists()
                
                # If there's no pending approval for this medication, add the interaction
                if not pending_exists:
                    interactions.append(r)
    
    # Remove duplicates
    seen = set()
    unique_interactions = []
    for item in interactions:
        key = f"{item.get('drug1', '')}_{item.get('drug2', '')}"
        if key not in seen:
            seen.add(key)
            unique_interactions.append(item)
    interactions = unique_interactions
    
    # Get dosage and time data
    dose_count = int(request.POST.get('dose_count', 1))
    times = []
    for i in range(1, dose_count + 1):
        time_str = request.POST.get(f'time_{i}', '').strip()
        if time_str:
            try:
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                times.append(time_obj)
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_str, '%I:%M %p').time()
                    times.append(time_obj)
                except ValueError:
                    pass
    
    start_date_str = request.POST.get('start_date')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else date.today()
    
    times_str = ', '.join([t.strftime('%H:%M') for t in times])
    first_time = times[0] if times else None
    compartment_number = int(request.POST.get('compartment_number', 0))
    
    # ===== IF INTERACTIONS FOUND, CREATE PENDING MEDICATION =====
    if interactions:
        approval_token = secrets.token_urlsafe(32)
        
        # Check if this medication already has a pending approval
        existing_pending = PendingMedication.objects.filter(
            name__iexact=request.POST.get('name'),
            status='pending'
        ).first()
        
        if existing_pending:
            messages.warning(request, 
                f"⚠️ {request.POST.get('name')} already has a pending approval request. "
                f"Please wait for caregiver approval.")
            return redirect('schedule')
        
        pending = PendingMedication.objects.create(
            name=request.POST.get('name'),
            generic_name=request.POST.get('generic_name', ''),
            category=request.POST.get('category', ''),
            instructions=request.POST.get('instructions', ''),
            dosage_count=int(request.POST.get('dosage_count', 1)),
            total_pills_loaded=int(request.POST.get('total_pills_loaded', 0)),
            low_quantity_threshold=int(request.POST.get('low_quantity_threshold', 5)),
            meal_instruction=request.POST.get('meal_instruction', 'any'),
            scheduled_time=first_time,
            start_date=start_date,
            multiple_times=times_str,
            interactions_found=json.dumps(interactions),
            approval_token=approval_token,
            status='pending'
        )
        
        # Store in session for the pending page
        request.session['pending_medication_id'] = pending.id
        request.session['interactions'] = interactions
        
        # Build approval URLs
        approval_url = request.build_absolute_uri(
            reverse('approve_medication', args=[approval_token])
        )
        denial_url = request.build_absolute_uri(
            reverse('deny_medication', args=[approval_token])
        )
        
        interaction_summary = "\n".join([
            f"- {i['drug1']} + {i['drug2']}: {i['description'][:100]}..."
            for i in interactions
        ])
        
        # Send email to caregiver
        if getattr(settings, 'CAREGIVER_EMAIL', None):
            try:
                send_mail(
                    subject=f'[SmartCare] Medication Approval Required: {request.POST.get("name")}',
                    message=f"""
A new medication ({request.POST.get('name')}) has been requested but has potential interactions with existing medications.

INTERACTIONS DETECTED:
{interaction_summary}

Schedule Times: {times_str if times_str else 'Not specified'}
Compartment: {compartment_number if compartment_number > 0 else 'Auto-assign'}

Please review and approve or deny this request:

✅ Approve: {approval_url}
❌ Deny: {denial_url}

This request will expire in 48 hours.
                    """,
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[settings.CAREGIVER_EMAIL],
                    fail_silently=True,
                )
            except Exception as e:
                print(f"❌ Email error: {e}")
        
        messages.warning(request, 
            f"⚠️ {request.POST.get('name')} has potential interactions. "
            f"An approval request has been sent to your caregiver.")
        
        return redirect('pending_approval')
    
    # ===== NO INTERACTIONS - SAVE DIRECTLY =====
    return save_medication_directly(request)


def save_medication_directly(request):
    """Save medication without caregiver approval - supports 1-3 daily times"""
    total_pills = int(request.POST.get('total_pills_loaded', 0))
    compartment_number = int(request.POST.get('compartment_number', 0))
    
    medication = Medication.objects.create(
        name=request.POST.get('name'),
        generic_name=request.POST.get('generic_name', ''),
        category=request.POST.get('category', ''),
        instructions=request.POST.get('instructions', ''),
        dosage_count=int(request.POST.get('dosage_count', 1)),
        total_pills_loaded=total_pills,
        pills_remaining=total_pills,
        low_quantity_threshold=int(request.POST.get('low_quantity_threshold', 5)),
        meal_instruction=request.POST.get('meal_instruction', 'any'),
        compartment_number=compartment_number,  # ← FIXED: Now saves compartment
    )
    
    start_date_str = request.POST.get('start_date')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else date.today()
    
    dose_count = int(request.POST.get('dose_count', 1))
    
    times = []
    for i in range(1, dose_count + 1):
        time_str = request.POST.get(f'time_{i}', '').strip()
        if time_str:
            try:
                time_obj = datetime.strptime(time_str, '%H:%M').time()
                times.append(time_obj)
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_str, '%I:%M %p').time()
                    times.append(time_obj)
                except ValueError:
                    messages.error(request, f"Invalid time format for Time {i}")
                    return redirect('schedule')
    
    if not times:
        messages.error(request, "Please enter at least one valid time.")
        return redirect('schedule')
    
    times.sort()
    
    created_count = 0
    for idx, time_obj in enumerate(times):
        Schedule.objects.create(
            medication=medication,
            scheduled_time=time_obj,
            start_date=start_date,
            status='active',
            dose_number=idx + 1,
            group_id=f"{medication.id}_{start_date}_{idx}"
        )
        created_count += 1
    
    messages.success(request, f"✅ {medication.name} added with {created_count} daily dose(s).")
    return redirect('schedule')


def pending_approval(request):
    """Show pending approval status page"""
    pending_id = request.session.get('pending_medication_id')
    interactions = request.session.get('interactions', [])
    
    if not pending_id:
        return redirect('schedule')
    
    try:
        pending = PendingMedication.objects.get(id=pending_id)
    except PendingMedication.DoesNotExist:
        return redirect('schedule')
    
    context = {
        'pending': pending,
        'interactions': interactions,
    }
    return render(request, 'pending_approval.html', context)


def approve_medication(request, token):
    """Caregiver approval endpoint - supports multiple times"""
    try:
        pending = PendingMedication.objects.get(approval_token=token, status='pending')
    except PendingMedication.DoesNotExist:
        messages.error(request, "Invalid or expired approval link.")
        return redirect('schedule')
    
    if request.method == 'POST':
        medication = Medication.objects.create(
            name=pending.name,
            generic_name=pending.generic_name,
            category=pending.category,
            instructions=pending.instructions,
            dosage_count=pending.dosage_count,
            total_pills_loaded=pending.total_pills_loaded,
            pills_remaining=pending.total_pills_loaded,
            low_quantity_threshold=pending.low_quantity_threshold,
            meal_instruction=pending.meal_instruction,
            compartment_number=0,  # Auto-assign for approved medications
        )
        
        start_date = pending.start_date
        
        times_str = pending.multiple_times
        times = []
        
        if times_str:
            for time_str in times_str.split(','):
                time_str = time_str.strip()
                if time_str:
                    try:
                        time_obj = datetime.strptime(time_str, '%H:%M').time()
                        times.append(time_obj)
                    except ValueError:
                        try:
                            time_obj = datetime.strptime(time_str, '%I:%M %p').time()
                            times.append(time_obj)
                        except ValueError:
                            pass
        
        if not times and pending.scheduled_time:
            times = [pending.scheduled_time]
        
        if not times:
            messages.error(request, "No valid schedule times found.")
            return redirect('schedule')
        
        times.sort()
        
        for idx, time_obj in enumerate(times):
            Schedule.objects.create(
                medication=medication,
                scheduled_time=time_obj,
                start_date=start_date,
                status='active',
                dose_number=idx + 1,
                group_id=f"{medication.id}_{start_date}_{idx}"
            )
        
        pending.status = 'approved'
        pending.approved_at = timezone.now()
        pending.save()
        
        request.session.pop('pending_medication_id', None)
        request.session.pop('interactions', None)
        request.session.pop('multiple_times', None)
        
        messages.success(request, f"✅ {pending.name} has been approved with {len(times)} daily dose(s).")
        return redirect('schedule')
    
    context = {
        'pending': pending,
        'interactions': json.loads(pending.interactions_found),
    }
    return render(request, 'approve_medication.html', context)


def deny_medication(request, token):
    """Caregiver denial endpoint"""
    try:
        pending = PendingMedication.objects.get(approval_token=token, status='pending')
    except PendingMedication.DoesNotExist:
        messages.error(request, "Invalid or expired denial link.")
        return redirect('schedule')
    
    pending.status = 'denied'
    pending.save()
    
    request.session.pop('pending_medication_id', None)
    request.session.pop('interactions', None)
    
    messages.info(request, f"❌ {pending.name} has been denied by caregiver.")
    return redirect('schedule')


def mark_taken(request, schedule_id):
    """Mark a pill as taken"""
    if request.method == 'POST':
        schedule = get_object_or_404(Schedule, id=schedule_id)
        
        if schedule.medication.dispense():
            schedule.intake_status = 'verified'
            schedule.last_verified_take = timezone.now()
            schedule.save()
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'success', 'message': 'Marked as taken'})
            
            messages.success(request, f"Marked {schedule.medication.name} as taken.")
            return redirect('schedule')
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'status': 'error', 'message': 'Not enough pills remaining'}, status=400)
            messages.error(request, f"Not enough pills remaining for {schedule.medication.name}.")
            return redirect('schedule')
    
    return redirect('schedule')


def get_schedule_api(request, date_str=None):
    """API endpoint to get schedule for a specific date"""
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    else:
        target_date = date.today()
    
    all_schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=target_date,
        medication__pills_remaining__gt=0
    ).select_related('medication')
    
    active_schedules = [s for s in all_schedules if s.is_active_today(target_date)]
    
    result = []
    for s in active_schedules:
        result.append({
            'id': s.id,
            'scheduled_time': s.scheduled_time.strftime('%H:%M'),
            'status': s.status,
            'medication_name': s.medication.name,
            'dosage_count': s.medication.dosage_count,
            'pills_remaining': s.medication.pills_remaining,
            'instructions': s.medication.instructions,
            'dose_number': s.dose_number,
        })
    
    return JsonResponse(result, safe=False)


# ==================== CLEAR OLD SCHEDULES ====================

@csrf_protect
def clear_old_schedules(request):
    """Clear schedules older than today"""
    if request.method == 'POST':
        today = date.today()
        old_schedules = Schedule.objects.filter(start_date__lt=today)
        count = old_schedules.count()
        old_schedules.delete()
        return JsonResponse({'status': 'success', 'message': f'Deleted {count} old schedules'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)


# ==================== NOTIFICATION STATUS API ====================

def get_notification_data_json(request):
    """API endpoint for notification status."""
    try:
        today = date.today()
        now = timezone.now()
        
        schedule = Schedule.objects.filter(
            status='active',
            start_date__lte=today,
            medication__pills_remaining__gt=0,
            intake_status='pending'
        ).order_by('scheduled_time').first()
        
        if schedule and schedule.is_active_today(today):
            scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
            scheduled_datetime = timezone.make_aware(scheduled_datetime)
            time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
            
            if -5 <= time_diff_minutes <= 60:
                return JsonResponse({'status': 'pending'})
            elif time_diff_minutes > 60:
                if schedule.intake_status == 'missed':
                    return JsonResponse({'status': 'missed'})
                return JsonResponse({'status': 'pending'})
        
        verified_today = Schedule.objects.filter(
            status='active',
            start_date__lte=today,
            intake_status='verified'
        ).exists()
        
        if verified_today:
            return JsonResponse({'status': 'verified'})
        
        return JsonResponse({'status': 'none'})
        
    except Exception as e:
        print(f"Error in notification status: {e}")
        return JsonResponse({'status': 'error'})
    
    
# ==================== ARDUINO API ENDPOINTS ====================
@csrf_exempt
def arduino_get_schedule(request):
    """
    API endpoint for ESP32 to fetch schedule.
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=405)
    
    today = date.today()
    now = timezone.now()
    
    schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication').order_by('scheduled_time')
    
    today_schedules = [s for s in schedules if s.is_active_today(today)]
    
    schedule_list = []
    for s in today_schedules:
        compartment = getattr(s.medication, 'compartment_number', 1)
        if compartment == 0:
            compartment = 1
        
        schedule_list.append({
            'time': s.scheduled_time.strftime('%H:%M'),
            'medication_name': s.medication.name,
            'dosage_count': s.medication.dosage_count,
            'compartment': compartment,
            'pills_remaining': s.medication.pills_remaining,
        })
    
    return JsonResponse({
        'status': 'success',
        'date': today.strftime('%Y-%m-%d'),
        'server_time': now.strftime('%H:%M:%S'),
        'schedule': schedule_list,
        'count': len(schedule_list)
    })
    
@csrf_exempt
def arduino_dispense_confirmation(request):
    """
    API endpoint for Arduino/ESP32 to confirm pills were dispensed.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        compartment = data.get('compartment')
        medication_name = data.get('medication_name')
        dosage_count = data.get('dosage_count', 1)
        
        today = date.today()
        schedule = Schedule.objects.filter(
            status='active',
            start_date__lte=today,
            medication__name__icontains=medication_name,
        ).first()
        
        if not schedule:
            return JsonResponse({
                'status': 'error',
                'message': f'Medication not found: {medication_name}'
            }, status=404)
        
        if schedule.medication.dispense():
            schedule.intake_status = 'verified'
            schedule.last_verified_take = timezone.now()
            schedule.save()
            
            IntakeLog.objects.create(
                schedule=schedule,
                medication=schedule.medication,
                scheduled_time=schedule.scheduled_time,
                actual_take_time=timezone.now(),
                date=today,  # ← ADD THIS
                status='on_time',
                is_verified=True,
                verification_method='arduino',
                notes=f'Dispensed by Arduino from compartment {compartment}'
            )
            
            return JsonResponse({
                'status': 'success',
                'message': f'Dispensed {medication_name}',
                'pills_remaining': schedule.medication.pills_remaining
            })
        else:
            return JsonResponse({
                'status': 'error',
                'message': f'Not enough pills for {medication_name}'
            }, status=400)
            
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def arduino_status_report(request):
    """
    API endpoint for Arduino/ESP32 to report system status.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        print(f"📊 Arduino Status Report: {data}")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Status report received'
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt
def test_command(request):
    """Test sending a command to ESP32"""
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=405)
    
    command = request.GET.get('cmd', 'STATUS')
    
    success, response = send_to_esp32(command)
    
    return JsonResponse({
        'status': 'success' if success else 'error',
        'command': command,
        'response': response
    })
    
@csrf_exempt
def esp32_get_schedule(request):
    """
    API endpoint for ESP32 to fetch the complete medication schedule.
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'GET required'}, status=405)
    
    today = date.today()
    now = timezone.now()
    
    schedules = Schedule.objects.filter(
        status='active',
        start_date__lte=today,
        medication__pills_remaining__gt=0
    ).select_related('medication').order_by('scheduled_time')
    
    today_schedules = [s for s in schedules if s.is_active_today(today)]
    
    schedule_list = []
    for s in today_schedules:
        compartment = getattr(s.medication, 'compartment_number', 1)
        if compartment == 0:
            compartment = 1
            
        schedule_list.append({
            'id': s.id,
            'time': s.scheduled_time.strftime('%H:%M'),
            'medication_name': s.medication.name,
            'dosage_count': s.medication.dosage_count,
            'compartment': compartment,
            'pills_remaining': s.medication.pills_remaining,
        })
    
    return JsonResponse({
        'status': 'success',
        'date': today.strftime('%Y-%m-%d'),
        'server_time': now.strftime('%H:%M:%S'),
        'schedule': schedule_list,
        'count': len(schedule_list)
    })
    
    
def send_to_esp32(command):
    """Send command to ESP32 if connected"""
    connected, esp = get_esp32_connection()
    
    if not connected or esp is None:
        return False, "ESP32 not connected"
    
    try:
        print(f"📤 Sending to ESP32: {command}")  # ← ADD THIS
        esp.write(f"{command}\n".encode())
        time.sleep(0.3)
        
        response = ""
        if esp.in_waiting > 0:
            response = esp.readline().decode().strip()
            print(f"📥 ESP32 response: {response}")  # ← ADD THIS
        
        return True, response if response else "Command sent successfully"
    except Exception as e:
        return False, str(e)
    
@csrf_exempt
def mark_missed(request, schedule_id):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    schedule = get_object_or_404(Schedule, id=schedule_id)
    today = date.today()
    now = timezone.now()
    
    if schedule.intake_status == 'verified':
        return JsonResponse({'status': 'error', 'message': 'Already verified'})
    
    if schedule.intake_status == 'missed':
        return JsonResponse({'status': 'error', 'message': 'Already marked as missed'})
    
    today_logs = IntakeLog.objects.filter(schedule=schedule, date=today)
    if today_logs.exists():
        return JsonResponse({'status': 'error', 'message': 'Log already exists'})
    
    scheduled_datetime = datetime.combine(today, schedule.scheduled_time)
    scheduled_datetime = timezone.make_aware(scheduled_datetime)
    time_diff_minutes = (now - scheduled_datetime).total_seconds() / 60
    
    schedule.intake_status = 'missed'
    schedule.save()
    
    IntakeLog.objects.create(
        schedule=schedule,
        medication=schedule.medication,
        scheduled_time=scheduled_datetime,
        date=today,
        status='missed',
        is_verified=False,
        minutes_late=int(time_diff_minutes) if time_diff_minutes > 0 else 0,
        notes='Auto-marked as missed after 15 minutes.'
    )
    
    return JsonResponse({
        'status': 'success',
        'message': f'Marked {schedule.medication.name} as missed'
    })
    
    
@csrf_exempt
def keep_pill(request, token, schedule_id):
    """Caregiver chooses to keep the pill (count stays the same)"""
    try:
        pending = PendingMissedPill.objects.get(
            approval_token=token,
            schedule_id=schedule_id,
            status='pending'
        )
    except PendingMissedPill.DoesNotExist:
        messages.error(request, "Invalid or expired link.")
        return redirect('schedule')
    
    if request.method == 'POST':
        # Keep the pill - pill count stays the same
        # The pill remains in the compartment, so we don't reduce pills_remaining
        pending.status = 'keep'
        pending.decision_made_at = timezone.now()
        pending.notes = request.POST.get('notes', '')
        pending.save()
        
        # Update the schedule status
        schedule = pending.schedule
        schedule.intake_status = 'pending'  # Reset so it can be taken later
        schedule.save()
        
        # Update the intake log
        IntakeLog.objects.filter(
            schedule=schedule,
            date=date.today(),
            status='missed'
        ).update(
            notes='Caregiver chose to keep the pill. Pill remains in compartment.',
            status='pending'
        )
        
        messages.success(request, f"✅ Pill kept. {pending.medication.name} count remains unchanged.")
        return redirect('schedule')
    
    context = {
        'pending': pending,
        'action': 'keep',
        'action_text': 'Keep the pill (count stays the same)',
        'button_color': 'green'
    }
    return render(request, 'pending_missed_decision.html', context)


@csrf_exempt
def discard_pill(request, token, schedule_id):
    """Caregiver chooses to discard the pill (count is reduced)"""
    try:
        pending = PendingMissedPill.objects.get(
            approval_token=token,
            schedule_id=schedule_id,
            status='pending'
        )
    except PendingMissedPill.DoesNotExist:
        messages.error(request, "Invalid or expired link.")
        return redirect('schedule')
    
    if request.method == 'POST':
        # Discard the pill - reduce pill count
        pending.status = 'discard'
        pending.decision_made_at = timezone.now()
        pending.notes = request.POST.get('notes', '')
        pending.save()
        
        # Reduce pills_remaining
        schedule = pending.schedule
        medication = schedule.medication
        
        # Remove the dispensed pill count
        if medication.pills_remaining >= medication.dosage_count:
            medication.pills_remaining -= medication.dosage_count
            medication.save()
        else:
            # If not enough pills, just set to 0
            medication.pills_remaining = 0
            medication.save()
        
        # Update the schedule status
        schedule.intake_status = 'missed'
        schedule.save()
        
        # Update the intake log
        IntakeLog.objects.filter(
            schedule=schedule,
            date=date.today(),
            status='missed'
        ).update(
            notes='Caregiver chose to discard the pill. Pill count reduced.',
            status='missed'
        )
        
        messages.success(request, f"✅ Pill discarded. {pending.medication.name} pill count reduced.")
        return redirect('schedule')
    
    context = {
        'pending': pending,
        'action': 'discard',
        'action_text': 'Discard the pill (count will be reduced)',
        'button_color': 'red'
    }
    return render(request, 'pending_missed_decision.html', context)



# ==================== EMOTION MONITOR VIEWS ====================

from .emotion_monitor import BehaviorMonitor, MockDataGenerator
import base64

# Initialize behavior monitor
behavior_monitor = BehaviorMonitor()

@csrf_exempt
def emotion_monitor_panel(request):
    """Display emotion monitoring dashboard"""
    context = {
        'current_emotion': behavior_monitor.emotion_recognizer.current_emotion,
        'emotion_confidence': behavior_monitor.emotion_recognizer.confidence,
        'emotion_history': list(behavior_monitor.emotion_recognizer.emotion_history)[-20:],
    }
    return render(request, 'emotion_monitor.html', context)

@csrf_exempt
def process_emotion_frame(request):
    """Process frame for emotion detection"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    try:
        import base64
        import cv2
        import numpy as np
        import json
        from datetime import datetime
        
        data = json.loads(request.body)
        image_data = data.get('image')
        adherence_data = data.get('adherence', {})
        
        if not image_data:
            return JsonResponse({'status': 'error', 'message': 'Missing image'}, status=400)
        
        # Decode image
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data.replace('data:image/jpeg;base64,', '')
        elif image_data.startswith('data:image/png;base64,'):
            image_data = image_data.replace('data:image/png;base64,', '')
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JsonResponse({'status': 'error', 'message': 'Invalid image'}, status=400)
        
        # Initialize behavior monitor if not already
        global behavior_monitor
        if 'behavior_monitor' not in globals() or behavior_monitor is None:
            from .emotion_monitor import BehaviorMonitor
            behavior_monitor = BehaviorMonitor()
        
        # Update monitor with new data
        features = behavior_monitor.update(frame, adherence_data)
        
        # Detect anomaly
        anomaly = behavior_monitor.detect_anomaly()
        
        # Helper function to convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        # If anomaly detected, send alert
        alert_sent = False
        if anomaly:
            try:
                from django.core.mail import send_mail
                from django.conf import settings
                
                subject = f'[SmartCare] Behavioral Anomaly Detected!'
                message = f"""
SMART CARE ALERT - BEHAVIORAL ANOMALY DETECTED

Time: {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Anomaly Score: {float(anomaly['anomaly_score']):.2f} (Threshold: 0.60)

Current Emotion: {anomaly.get('current_emotion', 'unknown')}
Recent Adherence: {float(anomaly.get('recent_adherence', 0)):.1f}%
Alert Count: {int(anomaly.get('alert_count', 0))}

This alert indicates unusual behavior patterns that may require attention.
Please check on the patient and review their medication adherence.

---
SmartCare Behavioral Monitoring System
"""
                send_mail(
                    subject=subject,
                    message=message,
                    from_email=settings.EMAIL_HOST_USER,
                    recipient_list=[settings.CAREGIVER_EMAIL],
                    fail_silently=False,
                )
                alert_sent = True
                print(f"Anomaly alert email sent")
            except Exception as e:
                print(f"Alert email failed: {e}")
            
            # Prepare response with converted values
            response_data = {
                'status': 'success',
                'emotion': str(features.get('emotion', 'neutral')),
                'confidence': float(features.get('confidence', 0.0)),
                'anomaly_detected': True,
                'anomaly_score': float(anomaly.get('anomaly_score', 0.0)),
                'alert_sent': alert_sent,
                'message': 'Anomaly detected! Alert sent to caregiver.' if alert_sent else 'Anomaly detected! (Email failed)'
            }
            return JsonResponse(convert_to_serializable(response_data))
        
        # Normal response - convert emotion history
        emotion_history = []
        for e in list(behavior_monitor.emotion_recognizer.emotion_history)[-20:]:
            emotion_history.append({
                'emotion': str(e.get('emotion', 'neutral')),
                'timestamp': e.get('timestamp').isoformat() if e.get('timestamp') else None,
                'confidence': float(e.get('confidence', 0.0))
            })
        
        response_data = {
            'status': 'success',
            'emotion': str(features.get('emotion', 'neutral')),
            'confidence': float(features.get('confidence', 0.0)),
            'anomaly_detected': False,
            'emotion_history': emotion_history
        }
        
        return JsonResponse(convert_to_serializable(response_data))
        
    except Exception as e:
        print(f"Emotion processing error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    
def send_anomaly_alert(anomaly_data):
    """Send email alert for detected anomaly"""
    try:
        subject = f'[SmartCare] Behavioral Anomaly Detected!'
        
        # In views.py, update the email message part:

        message = f"""
        SMART CARE ALERT - BEHAVIORAL ANOMALY DETECTED

        Time: {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        Anomaly Score: {float(anomaly['anomaly_score']):.2f}

        Current Emotion: {anomaly.get('current_emotion', 'unknown')}
        Recent Adherence: {float(anomaly.get('recent_adherence', 0)):.1f}%
        Alert Count: {int(anomaly.get('alert_count', 0))}

        NEGATIVE EMOTION DETAILS:
        Duration: {anomaly.get('negative_duration_minutes', 0):.1f} minutes
        Negative Emotion Percentage: {anomaly.get('negative_percentage', 0):.1f}%

        This alert indicates the user has been showing negative emotions for 
        an extended period (more than 5 minutes) with a high percentage of 
        negative emotions.

        Action Items:
        1. Check on the patient's emotional state
        2. Verify medication adherence
        3. Check if medication dispenser is working properly
        4. Contact patient if necessary

        ---
        SmartCare Behavioral Monitoring System
        """
        
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[settings.CAREGIVER_EMAIL],
            fail_silently=False,
        )
        print(f"Anomaly alert email sent to {settings.CAREGIVER_EMAIL}")
        
    except Exception as e:
        print(f"Failed to send alert email: {e}")
        
        
# views.py - Add these view functions

from .models import BehavioralData, BehaviorReport
from .report_generator import ReportGenerator
import json
from django.db.models import Count, Avg
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect

# Initialize report generator
report_generator = ReportGenerator()

@csrf_exempt
def generate_synthetic_data(request):
    """Generate 2 months of synthetic data with anomalies"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    try:
        # Always generate exactly 60 days (2 months)
        days = 60
        samples_per_day = 24  # One sample per hour
        
        # Generate synthetic data with anomalies
        data = report_generator.generate_synthetic_data(
            days=days, 
            samples_per_day=samples_per_day
        )
        
        # Save to database
        saved_count = 0
        anomaly_count = 0
        
        for record in data:
            BehavioralData.objects.create(
                timestamp=record['timestamp'],
                emotion=record['emotion'],
                emotion_score=record['emotion_score'],
                confidence=record['confidence'],
                adherence=record['adherence'],
                missed_pills=record['missed_pills'],
                stress_level=record['stress_level'],
                behavior_score=record['behavior_score'],
                is_synthetic=True,
                is_anomaly=record.get('is_anomaly', False)
            )
            saved_count += 1
            if record.get('is_anomaly', False):
                anomaly_count += 1
        
        return JsonResponse({
            'status': 'success',
            'message': f'✅ Loaded {saved_count} records for 2 months with {anomaly_count} anomalies',
            'count': saved_count,
            'anomaly_count': anomaly_count,
            'days': days
        })
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@csrf_exempt
def generate_report(request):
    """Generate a behavior report and send email"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=400)
    
    try:
        data = json.loads(request.body)
        report_type = data.get('report_type', 'daily')
        days = int(data.get('days', 7))
        send_email = data.get('send_email', True)
        
        # Get date range
        end_date = timezone.now()
        start_date = end_date - timedelta(days=days)
        
        # Get behavioral data
        behavioral_records = BehavioralData.objects.filter(
            timestamp__gte=start_date,
            timestamp__lte=end_date
        ).order_by('timestamp')
        
        if not behavioral_records.exists():
            return JsonResponse({
                'status': 'error',
                'message': 'No data available for the selected period. Generate synthetic data first.'
            }, status=400)
        
        # Convert to list of dicts
        records_data = []
        for record in behavioral_records:
            records_data.append({
                'timestamp': record.timestamp,
                'emotion': record.emotion,
                'emotion_score': record.emotion_score,
                'confidence': record.confidence,
                'adherence': record.adherence,
                'missed_pills': record.missed_pills,
                'stress_level': record.stress_level,
                'behavior_score': record.behavior_score
            })
        
        # Generate report
        report = report_generator.generate_report(start_date, end_date, records_data)
        
        if not report:
            return JsonResponse({
                'status': 'error',
                'message': 'Failed to generate report'
            }, status=500)
        
        # Save report to database
        report_obj = BehaviorReport.objects.create(
            report_type=report_type,
            date_range_start=start_date,
            date_range_end=end_date,
            total_data_points=report['total_records'],
            average_emotion_score=report['average_emotion_score'],
            average_adherence=report['average_adherence'],
            total_missed_pills=report['total_missed_pills'],
            anomaly_count=report['anomaly_count'],
            max_anomaly_score=report['max_anomaly_score'],
            dominant_emotion=report['dominant_emotion'],
            emotion_distribution=report['emotion_distribution'],
            summary=report['summary'],
            recommendations=report['recommendations']
        )
        
        # Send email if requested
        email_sent = False
        if send_email and settings.CAREGIVER_EMAIL:
            try:
                send_report_email(report, start_date, end_date, settings.CAREGIVER_EMAIL)
                report_obj.email_sent = True
                report_obj.email_sent_at = timezone.now()
                report_obj.email_recipient = settings.CAREGIVER_EMAIL
                report_obj.save()
                email_sent = True
            except Exception as e:
                print(f"Email error: {e}")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Report generated successfully!',
            'report_id': report_obj.id,
            'email_sent': email_sent,
            'report': {
                'summary': report['summary'],
                'recommendations': report['recommendations'],
                'average_adherence': f"{report['average_adherence']*100:.1f}%",
                'anomaly_count': report['anomaly_count'],
                'dominant_emotion': report['dominant_emotion'],
            }
        })
        
    except Exception as e:
        print(f"Report error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def report_dashboard(request):
    """Display the report dashboard"""
    recent_reports = BehaviorReport.objects.all()[:10]
    
    # Get stats
    total_records = BehavioralData.objects.count()
    synthetic_records = BehavioralData.objects.filter(is_synthetic=True).count()
    real_records = total_records - synthetic_records
    anomaly_count = BehavioralData.objects.filter(is_anomaly=True).count()
    
    # Get recent data for chart
    recent_data = BehavioralData.objects.order_by('-timestamp')[:100]
    
    context = {
        'recent_reports': recent_reports,
        'total_records': total_records,
        'synthetic_records': synthetic_records,
        'real_records': real_records,
        'anomaly_count': anomaly_count,
        'recent_data': recent_data,
    }
    return render(request, 'report_dashboard.html', context)


def send_report_email(report, start_date, end_date, recipient):
    """Send report via email"""
    try:
        # Create HTML email
        subject = f'[SmartCare] Behavioral Report - {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
        
        emoji_map = {
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'angry': '😡',
            'fear': '😨',
            'surprise': '😮',
            'disgust': '🤢'
        }
        
        dominant_emoji = emoji_map.get(report['dominant_emotion'], '😐')
        
        # Build emotion distribution
        emotion_lines = []
        for emotion, count in sorted(report['emotion_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            emoji = emoji_map.get(emotion, '')
            emotion_lines.append(f"  {emoji} {emotion.title()}: {count}")
        
        html_message = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; border-bottom: 2px solid #2563eb; padding-bottom: 20px; }}
                .header h1 {{ color: #2563eb; margin: 0; }}
                .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
                .stat {{ background: #f8fafc; padding: 15px; border-radius: 8px; text-align: center; }}
                .stat .value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
                .stat .label {{ font-size: 12px; color: #64748b; margin-top: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f8fafc; border-radius: 8px; }}
                .section h3 {{ margin: 0 0 10px 0; color: #1e293b; }}
                .anomaly-warning {{ background: #fee2e2; border-left: 4px solid #dc2626; padding: 15px; margin: 15px 0; border-radius: 8px; }}
                .footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
                ul {{ margin: 5px 0; padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 SmartCare Behavioral Report</h1>
                    <p style="color: #64748b;">
                        {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}
                    </p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="value">{dominant_emoji} {report['dominant_emotion'].title()}</div>
                        <div class="label">Dominant Emotion</div>
                    </div>
                    <div class="stat">
                        <div class="value">{report['average_adherence']*100:.1f}%</div>
                        <div class="label">Average Adherence</div>
                    </div>
                    <div class="stat">
                        <div class="value">{report['total_missed_pills']}</div>
                        <div class="label">Total Missed Pills</div>
                    </div>
                    <div class="stat">
                        <div class="value">{report['anomaly_count']}</div>
                        <div class="label">Anomalies Detected</div>
                    </div>
                </div>
                
                <div class="section">
                    <h3>📊 Summary</h3>
                    <p>{report['summary']}</p>
                </div>
                
                <div class="section">
                    <h3>📈 Emotion Distribution</h3>
                    <pre style="margin: 0; white-space: pre-wrap; font-family: monospace;">{chr(10).join(emotion_lines)}</pre>
                </div>
                
                <div class="section">
                    <h3>💡 Recommendations</h3>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in report['recommendations'].split('\\n') if rec)}
                    </ul>
                </div>
                
                {f'''
                <div class="anomaly-warning">
                    <strong>⚠️ Anomalies Detected</strong>
                    <p style="margin: 5px 0 0 0;">{report['anomaly_count']} anomalies were detected during this period. 
                    Max anomaly score: {report['max_anomaly_score']:.2f}</p>
                </div>
                ''' if report['anomaly_count'] > 0 else ''}
                
                <div class="footer">
                    <p>This report was generated automatically by SmartCare.</p>
                    <p>LSTM Autoencoder Status: {'✅ Trained' if report['lstm_trained'] else '⚠️ Not trained'}</p>
                    <p>For medical concerns, please consult your healthcare provider.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        send_mail(
            subject=subject,
            message=report['summary'],
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[recipient],
            html_message=html_message,
            fail_silently=False,
        )
        print(f"✅ Report email sent to {recipient}")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise
    
def settings_panel(request):
    """ESP32 Control Panel - Settings page"""
    return render(request, 'settings.html')

def intake_test_panel(request):
    """Intake Verification Test Panel"""
    return render(request, 'intake_test.html')