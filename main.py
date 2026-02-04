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

def get_counts():
    """Reads the CSV and returns unique session counts per gesture."""
    counts = { 'neutral': 0, 'smile': 0, 'frown': 0, 'blink': 0, 'leftWink': 0, 'rightWink': 0 }
    if not os.path.exists('test_case.csv'): return counts
    try:
        with open('test_case.csv', 'r') as f:
            reader = csv.DictReader(f)
            sessions = {}
            for row in reader:
                g, s_id = row['target_gesture'], row['session_id']
                if g not in sessions: sessions[g] = set()
                sessions[g].add(s_id)
            for g in sessions:
                if g in counts: counts[g] = len(sessions[g])
    except: pass
    return counts

def save_to_csv(dS, l, u_id):
    filename = 'test_case.csv'
    file_exists = os.path.isfile(filename)
    session_id = int(time.time())
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if not file_exists: writer.writeheader()
        for row in dS:
            row_to_write = row.copy()
            row_to_write['target_gesture'], row_to_write['session_id'], row_to_write['user_id'] = l, session_id, u_id
            writer.writerow(row_to_write)
    print(f"✅ Saved session: {l}")

def main():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    cap = cv.VideoCapture(0)
    
    # --- ALL VARIABLES INITIALIZED HERE ---
    tracked_indices = [1, 33, 263, 61, 291, 152, 159, 145, 386, 374, 55, 285, 13, 14]
    dequeStorage = deque(maxlen=150)
    label, user_id = 'neutral', 'me'
    night_mode = False
    last_timestamp_ms = 0  # Fixed: Defined before the loop
    gesture_counts = get_counts()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv.flip(frame, 1)
            display_frame = frame.copy()
            h, w, _ = display_frame.shape

            if night_mode:
                display_frame.fill(0)

            # --- MONOTONIC TIMESTAMP FIX ---
            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, current_timestamp_ms)

            if result.face_landmarks:
                face = result.face_landmarks[0]
                ear = abs(face[159].y - face[145].y)

                # UI OVERLAYS
                cv.rectangle(display_frame, (0, 0), (w, 100), (0, 0, 0), -1)
                cv.rectangle(display_frame, (w-200, 100), (w, 320), (0, 0, 0), -1)
                
                cv.putText(display_frame, f"GESTURE: {label.upper()}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                buf_color = (0, 255, 0) if len(dequeStorage) == 150 else (0, 0, 255)
                cv.putText(display_frame, f"BUFFER: {len(dequeStorage)}/150", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.9, buf_color, 2)
                cv.putText(display_frame, f"EYE: {ear:.4f}", (20, h-20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Sidebar Counters
                y_off = 130
                for g, count in gesture_counts.items():
                    y_off += 30
                    color = (0, 255, 0) if g == label else (255, 255, 255)
                    cv.putText(display_frame, f"{g}: {count}", (w-180, y_off), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                if len(tracked_indices) > 0:
                    frame_data = {}
                    nose, eye_l, eye_r = face[1], face[33], face[263]
                    dist = np.sqrt((eye_r.x-eye_l.x)**2 + (eye_r.y-eye_l.y)**2 + (eye_r.z-eye_l.z)**2)
                    dist = dist if dist > 0 else 1.0

                    for i, idx in enumerate(tracked_indices):
                        lm = face[idx]
                        frame_data[f"{LANDMARK_NAMES[i]}_x"] = (lm.x - nose.x) / dist
                        frame_data[f"{LANDMARK_NAMES[i]}_y"] = (lm.y - nose.y) / dist
                        frame_data[f"{LANDMARK_NAMES[i]}_z"] = (lm.z - nose.z) / dist
                        cv.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                    dequeStorage.append(frame_data)

            cv.imshow('Data Collector', display_frame)

            key = cv.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('n'): night_mode = not night_mode
            elif key == ord('c'):
                dequeStorage.clear()
                tracked_indices = []
            elif key == ord('r'):
                tracked_indices = [1, 33, 263, 61, 291, 152, 159, 145, 386, 374, 55, 285, 13, 14]
            elif key == ord('0'): label = 'neutral'
            elif key == ord('1'): label = 'smile'
            elif key == ord('2'): label = 'frown'
            elif key == ord('3'): label = 'blink'
            elif key == ord('4'): label = 'leftWink'
            elif key == ord('5'): label = 'rightWink'
            elif key == ord('o'):
                if len(dequeStorage) == 150:
                    save_to_csv(dequeStorage, label, user_id)
                    dequeStorage.clear()
                    gesture_counts = get_counts()
                else: print("Wait for buffer!")

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()