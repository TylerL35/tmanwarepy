import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load Model and Classes
model = tf.keras.models.load_model('gesture_model.keras')
class_names = np.load('classes.npy', allow_pickle=True)

# 2. Setup Tasks API Face Landmarker
model_path = 'face_landmarker.task' # Ensure this file is in your folder!
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

# Points used in your training:
TRACKED_INDICES = [1, 33, 263, 61, 291, 152, 159, 145, 386, 374, 55, 285, 13, 14]

cap = cv.VideoCapture(0)
landmarker = vision.FaceLandmarker.create_from_options(options)

print("🚀 Tasks API Predictor Online. Press 'ESC' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Tasks API requires a timestamp for VIDEO mode
    timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.face_landmarks:
        face = result.face_landmarks[0]
        
        # --- MATH: Must match your collector/trainer exactly ---
        features = []
        nose, eye_l, eye_r = face[1], face[33], face[263]
        dist = np.sqrt((eye_r.x-eye_l.x)**2 + (eye_r.y-eye_l.y)**2 + (eye_r.z-eye_l.z)**2)
        dist = dist if dist > 0 else 1.0

        for idx in TRACKED_INDICES:
            lm = face[idx]
            features.extend([
                (lm.x - nose.x) / dist,
                (lm.y - nose.y) / dist,
                (lm.z - nose.z) / dist
            ])
        
        # Predict
        input_data = np.array([features])
        prediction = model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        gesture = class_names[class_idx]

        # --- UI WITH BLACK BACKGROUND & BOLD TEXT ---
        text = f"{gesture.upper()} {confidence*100:.1f}%"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 3  # Increased thickness for a bold look
        pos = (20, 50)
        
        # 1. Calculate the background box size based on the text
        (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thickness)
        
        # 2. Draw the black rectangle (the background)
        # We add 10px padding for a professional "label" look
        cv.rectangle(frame, 
                     (pos[0] - 10, pos[1] - text_h - 10), 
                     (pos[0] + text_w + 10, pos[1] + baseline + 10), 
                     (0, 0, 0), 
                     -1) # -1 fills the box
        
        # 3. Draw the bold text on top
        if gesture == 'smile':
            color = (0, 255, 0) 
        elif gesture == 'frown':
            color = (0, 0, 255)
        else: 
            color = (255, 255, 255)
        cv.putText(frame, text, pos, font, font_scale, color, thickness)

    cv.imshow('MediaPipe Tasks Prediction', frame)
    if cv.waitKey(1) & 0xFF == 27: break

landmarker.close()
cap.release()
cv.destroyAllWindows()