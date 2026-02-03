import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import csv
import os
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Configuration
model_path = 'face_landmarker.task'
LANDMARK_NAMES = [
    'nose', 'eyeCornerL', 'eyeCornerR', 'mouthCornerL', 'mouthCornerR', 'chin',
    'eyelidLTop', 'eyelidLBot', 'eyelidRTop', 'eyelidRBot', 'browL', 'browR',
    'lipMidTop', 'lipMidBot'
]

FIELDNAMES = []
for name in LANDMARK_NAMES:
    FIELDNAMES.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
FIELDNAMES.extend(['target_gesture', 'session_id', 'user_id'])

def save_to_csv(dS, l, u_id):
    filename = 'test_case.csv'
    file_exists = os.path.isfile(filename)
    session_id = int(time.time())

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        
        for row in dS:
            row_to_write = row.copy()
            row_to_write['target_gesture'] = l
            row_to_write['session_id'] = session_id
            row_to_write['user_id'] = u_id
            writer.writerow(row_to_write)
            
    print(f"Appended 150 frames: {l} (User: {u_id})")

def main():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    cap = cv.VideoCapture(0)
    tracked_indices = []
    dequeStorage = deque(maxlen=150)
    label = 'neutral'
    user_id = 'me'

    print("Hot keys: \n\tESC - quit\n\tr - init landmarks\n\t0 - neutral\n\t1 - smile\n\t2 - frown\n\t3 - blink\n\t4 - leftWink\n\t5 - rightWink\n\to - SAVE 150 frames")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv.flip(frame, 1)
            display_frame = frame.copy()
            h, w, _ = display_frame.shape

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            if result.face_landmarks:
                face = result.face_landmarks[0]
                
                # Eye distance logic
                eye_top = face[159]
                eye_bot = face[145]
                ear = abs(eye_top.y - eye_bot.y)

                # 1. DRAW BACKGROUND BOXES FOR READABILITY
                cv.rectangle(display_frame, (0, 0), (300, 110), (0, 0, 0), -1) # Top left box

                # 2. DRAW STATUS TEXT
                # Current Gesture Label
                cv.putText(display_frame, f"Gesture: {label.upper()}", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Buffer Status (Red if filling, Green if full)
                buf_color = (0, 255, 0) if len(dequeStorage) == 150 else (0, 0, 255)
                cv.putText(display_frame, f"Buffer: {len(dequeStorage)}/150", (10, 65), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, buf_color, 2)
                
                if len(tracked_indices) > 0:
                    frame_data = {}
                    nose = face[1]
                    eye_l, eye_r = face[33], face[263]
                    dist = np.sqrt((eye_r.x-eye_l.x)**2 + (eye_r.y-eye_l.y)**2 + (eye_r.z-eye_l.z)**2)
                    dist = dist if dist > 0 else 1.0

                    for i, idx in enumerate(tracked_indices):
                        lm = face[idx]
                        frame_data[f"{LANDMARK_NAMES[i]}_x"] = (lm.x - nose.x) / dist
                        frame_data[f"{LANDMARK_NAMES[i]}_y"] = (lm.y - nose.y) / dist
                        frame_data[f"{LANDMARK_NAMES[i]}_z"] = (lm.z - nose.z) / dist
                        cv.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                    
                    dequeStorage.append(frame_data)

            cv.imshow('Safe MediaPipe Tracker', display_frame)

            key = cv.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('r'):
                tracked_indices = [1, 33, 263, 61, 291, 152, 159, 145, 386, 374, 55, 285, 13, 14]
            elif key == ord('0'): label = 'neutral'; print("Target: neutral")
            elif key == ord('1'): label = 'smile'; print("Target: smile")
            elif key == ord('2'): label = 'frown'; print("Target: frown")
            elif key == ord('3'): label = 'blink'; print("Target: blink")
            elif key == ord('4'): label = 'leftWink'; print("Target: leftWink")
            elif key == ord('5'): label = 'rightWink'; print("Target: rightWink")
            elif key == ord('o'):
                if len(dequeStorage) == 150:
                    save_to_csv(dequeStorage, label, user_id)
                else:
                    print(f"⚠️ Buffer only at {len(dequeStorage)}/150")

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()