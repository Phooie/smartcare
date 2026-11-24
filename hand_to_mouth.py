import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model files (run once)
import urllib.request

# Download Hand Landmarker model
hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
urllib.request.urlretrieve(hand_url, "hand_landmarker.task")

# Download Face Landmarker model  
face_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
urllib.request.urlretrieve(face_url, "face_landmarker.task")

# Initialize Hand Landmarker
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# Initialize Face Landmarker
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    num_faces=1,
    min_face_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# Mouth landmark indices in Face Landmarker (468 points)
# The inner mouth/lip area indices
# MOUTH_INDICES = [78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]




# Better symmetric mouth landmarks
MOUTH_INDICES = [
    13,   # upper lip center
    14,   # lower lip center
    78,   # left mouth corner
    308,  # right mouth corner
    81,   # upper left lip
    311,  # upper right lip
    178,  # lower left lip
    402   # lower right lip
]

INDEX_TIP = 8  # Index finger tip in Hand Landmarker (21 points)

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Bring your hand near your mouth...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands and face
    hand_result = hand_landmarker.detect(mp_image)
    face_result = face_landmarker.detect(mp_image)
    
    mouth_center = None
    
    # Get mouth center from face landmarks
    if face_result.face_landmarks:
        h, w = frame.shape[:2]
        face_landmarks = face_result.face_landmarks[0]

        left_corner = face_landmarks[78]
        right_corner = face_landmarks[308]
        upper_lip = face_landmarks[13]
        lower_lip = face_landmarks[14]

        mouth_center = (
            int(((left_corner.x + right_corner.x) / 2) * w),
            int(((upper_lip.y + lower_lip.y) / 2) * h)
        )

        cv2.circle(frame, mouth_center, 6, (0, 255, 255), -1)

        cv2.putText(
            frame,
            "MOUTH",
            (mouth_center[0] - 20, mouth_center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
    
    # Check hand distance to mouth
    pill_detected = False
    if hand_result.hand_landmarks and mouth_center:
        h, w = frame.shape[:2]
        for hand_landmarks in hand_result.hand_landmarks:
            index_tip = hand_landmarks[INDEX_TIP]
            finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
            
            distance = np.sqrt((finger_pos[0] - mouth_center[0])**2 + 
                              (finger_pos[1] - mouth_center[1])**2)
            
            # Draw line from finger to mouth
            color = (0, 0, 255) if distance < 50 else (0, 255, 0)
            cv2.line(frame, finger_pos, mouth_center, color, 2)
            cv2.circle(frame, finger_pos, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"Distance: {int(distance)}px", 
                       (finger_pos[0] - 30, finger_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            threshold = w * 0.08
            if distance < threshold:  # Threshold for pill-taking
                pill_detected = True
                print("💊 PILL TAKING GESTURE DETECTED!")
    
    # Display status
    if pill_detected:
        status = "STATUS: PILL TAKING DETECTED!"
        color = (0, 255, 0)
    elif mouth_center:
        status = "STATUS: PLEASE TAKE YOUR PILL"
        color = (0, 255, 0)
    else:
        status = "STATUS: Looking for face..."
        color = (255, 255, 255)
    
    cv2.putText(frame, status, (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow('Hand-to-Mouth Pill Taking Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()