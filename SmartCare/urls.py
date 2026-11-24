# from django.contrib import admin
# from django.urls import path
# from . import views

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', views.main, name='main'),
#     path('schedule/', views.schedule, name='schedule'),
#     path('intake-verification/', views.intake_verification, name='intake_verification'),
#     path('detect-gesture/', views.detect_gesture, name='detect_gesture'),  # For real-time detection
#     path('verify-intake/', views.verify_intake, name='verify_intake'),      # For manual verification
#     path('add_medication/', views.add_medication, name='add_medication'),
#     path('mark-taken/<int:schedule_id>/', views.mark_taken, name='mark_taken'),
#     path('api/schedule/<str:date_str>/', views.get_schedule_api, name='api_schedule'),
#     path('pending-approval/', views.pending_approval, name='pending_approval'),
#     path('approve/<str:token>/', views.approve_medication, name='approve_medication'),
#     path('deny/<str:token>/', views.deny_medication, name='deny_medication'),
#     path('request-remove/<int:schedule_id>/', views.request_remove_medication, name='request_remove'),
#     path('approve-removal/<str:token>/<int:schedule_id>/', views.approve_removal, name='approve_removal'),
#     path('deny-removal/<str:token>/<int:schedule_id>/', views.deny_removal, name='deny_removal'),
#     # # path('api/notification-status/', views.get_notification_data_json, name='notification_status'),
#     path('trigger-dose/<int:schedule_id>/', views.trigger_medication_dose, name='trigger_dose'),
#     path('close-compartments/', views.close_compartments, name='close_compartments'),
#     path('test-esp32/', views.test_esp32, name='test_esp32'),
#     path('get-detection-status/', views.get_detection_status, name='get_detection_status'),
#     path('api/schedule/<str:date_str>/', views.get_schedule_api, name='api_schedule'),
#     path('api/schedule/', views.get_schedule_api, name='api_schedule'),
#     path('api/notification-status/', views.get_notification_data_json, name='notification_status'),
# ]


from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # Main pages
    path('', views.main, name='main'),
    path('schedule/', views.schedule, name='schedule'),
    path('intake-verification/', views.intake_verification, name='intake_verification'),
    
    # ===== DETECTION ENDPOINTS =====
    path('detect-gesture/', views.detect_gesture, name='detect_gesture'),
    path('verify-intake/', views.verify_intake, name='verify_intake'),
    path('get-detection-status/', views.get_detection_status, name='get_detection_status'),
    
    # ===== ESP32 CONTROL =====
    path('test-esp32/', views.test_esp32, name='test_esp32'),
    path('trigger-dose/<int:schedule_id>/', views.trigger_medication_dose, name='trigger_dose'),
    path('close-compartments/', views.close_compartments, name='close_compartments'),
    
    # ===== ARDUINO API (for ESP32) - ADD THESE =====
    path('api/arduino/schedule/', views.arduino_get_schedule, name='arduino_schedule'),
    path('api/arduino/dispense/', views.arduino_dispense_confirmation, name='arduino_dispense'),
    path('api/arduino/status/', views.arduino_status_report, name='arduino_status'),
    
    # ===== MEDICATION FUNCTIONS =====
    path('add_medication/', views.add_medication, name='add_medication'),
    path('mark-taken/<int:schedule_id>/', views.mark_taken, name='mark_taken'),
    path('pending-approval/', views.pending_approval, name='pending_approval'),
    path('approve/<str:token>/', views.approve_medication, name='approve_medication'),
    path('deny/<str:token>/', views.deny_medication, name='deny_medication'),
    path('request-remove/<int:schedule_id>/', views.request_remove_medication, name='request_remove'),
    path('approve-removal/<str:token>/<int:schedule_id>/', views.approve_removal, name='approve_removal'),
    path('deny-removal/<str:token>/<int:schedule_id>/', views.deny_removal, name='deny_removal'),
    
    # ===== API ENDPOINTS =====
    path('api/schedule/<str:date_str>/', views.get_schedule_api, name='api_schedule'),
    path('api/schedule/', views.get_schedule_api, name='api_schedule'),
    path('api/notification-status/', views.get_notification_data_json, name='notification_status'),
    # Add this to urlpatterns
    path('test-cmd/', views.test_command, name='test_command'),
    # Arduino/ESP32 API
    path('api/esp32/schedule/', views.esp32_get_schedule, name='esp32_schedule'),

    # ===== EMOTION MONITOR =====
    path('emotion-monitor/', views.emotion_monitor_panel, name='emotion_monitor'),
    path('process-emotion-frame/', views.process_emotion_frame, name='process_emotion_frame'),
    path('report-dashboard/', views.report_dashboard, name='report_dashboard'),
    path('generate-synthetic-data/', views.generate_synthetic_data, name='generate_synthetic_data'),
    path('generate-report/', views.generate_report, name='generate_report'),
    path('settings/', views.settings_panel, name='settings'),
    path('intake-test/', views.intake_test_panel, name='intake_test'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)