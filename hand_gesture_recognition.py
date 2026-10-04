"""
Hand Gesture Recognition
------------------------
Uses MediaPipe's HandLandmarker (Tasks API) to detect hand landmarks
from a live webcam feed, classifies simple gestures (fist, open palm,
thumbs up, peace sign, pointing), draws landmarks/labels on the video
frame, and maps each gesture to a demo action (play/pause, volume, etc).

NOTE ON MEDIAPIPE VERSION: MediaPipe deprecated the old
`mediapipe.solutions.hands` API in recent releases in favor of the
Tasks API used below (`mediapipe.tasks.vision.HandLandmarker`). This
script uses the current, supported API so it keeps working with a
plain `pip install mediapipe`.

Run:
    pip install opencv-python mediapipe
    python3 hand_gesture_recognition.py
The hand-tracking model (~10 MB) is downloaded automatically on first
run. Press 'q' to quit the webcam window.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Landmark indices (21 points per hand, same numbering MediaPipe has always used)
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Skeleton connections for drawing (standard MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

# Gesture -> demo action mapping (printed to console as a stand-in
# for a real action like a media-key press or volume change).
GESTURE_ACTIONS = {
    "Open Palm": "PLAY",
    "Fist": "PAUSE",
    "Thumbs Up": "VOLUME UP",
    "Peace Sign": "NEXT TRACK",
    "Pointing": "MUTE",
}


def ensure_model_downloaded():
    """Download the HandLandmarker model bundle on first run if it's
    not already present next to this script.
    """
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand-tracking model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the hand-tracking model automatically "
            f"({exc}). Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {MODEL_PATH}"
        ) from exc


def _fingers_up(landmarks, handedness_label):
    """Return [thumb, index, middle, ring, pinky] booleans: True if extended.

    landmarks: list of (x, y) tuples in normalized image coordinates (0-1).
    handedness_label: "Left" or "Right" as reported by MediaPipe (the
        camera image is mirrored, so this flips which way is "extended"
        for the thumb).
    """
    fingers = []

    if handedness_label == "Right":
        fingers.append(landmarks[THUMB_TIP][0] < landmarks[THUMB_IP][0])
    else:
        fingers.append(landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0])

    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip][1] < landmarks[pip][1])

    return fingers


def classify_gesture(landmarks, handedness_label):
    """Classify a hand pose from its 21 landmarks into a named gesture.
    Returns "Unknown" if it doesn't match a recognized pattern.
    """
    thumb, index, middle, ring, pinky = _fingers_up(landmarks, handedness_label)
    up_count = sum([thumb, index, middle, ring, pinky])

    if up_count == 5:
        return "Open Palm"
    if up_count == 0:
        return "Fist"
    if thumb and not any([index, middle, ring, pinky]):
        return "Thumbs Up"
    if index and middle and not any([thumb, ring, pinky]):
        return "Peace Sign"
    if index and not any([thumb, middle, ring, pinky]):
        return "Pointing"
    return "Unknown"


def draw_hand(frame, landmarks_px):
    """Draw skeleton lines and landmark dots on the frame.
    landmarks_px: list of (x, y) pixel coordinates.
    """
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start], landmarks_px[end], (0, 200, 0), 2)
    for x, y in landmarks_px:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)


def run_webcam_demo(camera_index=0, cooldown_seconds=1.0):
    """Open the webcam, detect hands frame-by-frame, classify gestures,
    draw the results, and trigger a demo action when a gesture is held.
    """
    ensure_model_downloaded()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open webcam. Check camera_index or permissions.")
        return

    last_gesture = None
    last_trigger_time = 0.0
    start_time = time.time()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    points = [(lm.x, lm.y) for lm in hand_landmarks]
                    points_px = [(int(x * w), int(y * h)) for x, y in points]
                    label = handedness[0].category_name  # "Left"/"Right"

                    draw_hand(frame, points_px)
                    gesture = classify_gesture(points, label)

                    wrist_x, wrist_y = points_px[WRIST]
                    cv2.putText(
                        frame, gesture, (wrist_x - 40, wrist_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
                    )

                    now = time.time()
                    if gesture in GESTURE_ACTIONS and (
                        gesture != last_gesture or now - last_trigger_time > cooldown_seconds
                    ):
                        action = GESTURE_ACTIONS[gesture]
                        print(f"[{time.strftime('%H:%M:%S')}] Gesture: {gesture} -> Action: {action}")
                        last_gesture = gesture
                        last_trigger_time = now
            else:
                last_gesture = None

            cv2.putText(
                frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
