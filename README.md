# Syntecxhub_Hand_Gesture_Recognition

A Python hand gesture recognition tool built for the Syntecxhub AI Internship (Task 3, Project 2).

## Features
- **Real-time hand landmark detection** using MediaPipe's `HandLandmarker` (the current Tasks API — MediaPipe deprecated the old `solutions.hands` API, so this uses the supported replacement).
- **Gesture classification**: Open Palm, Fist, Thumbs Up, Peace Sign, Pointing — based on which fingers are extended.
- **Action mapping** (printed to console as a demo):
  | Gesture | Action |
  |---|---|
  | Open Palm | PLAY |
  | Fist | PAUSE |
  | Thumbs Up | VOLUME UP |
  | Peace Sign | NEXT TRACK |
  | Pointing | MUTE |
- **Live webcam overlay**: draws the hand skeleton and the detected gesture name on each frame.
- Works with **both hands** and handles multiple hands at once (`num_hands=2`).

## How to Run
```bash
pip install opencv-python mediapipe
python3 hand_gesture_recognition.py
```
- A small (~10 MB) hand-tracking model downloads automatically the first time you run it — this needs an internet connection once.
- A window opens showing your webcam feed with the hand skeleton drawn on top and the recognized gesture labeled near your wrist.
- Whenever you hold a recognized gesture, the mapped action prints to the terminal (e.g. `Gesture: Fist -> Action: PAUSE`).
- Press **`q`** to quit.

## How It Works
- `classify_gesture()` takes the 21 hand landmarks MediaPipe returns and checks which fingers are "up" (extended) by comparing each fingertip's position to its middle joint.
- The combination of extended fingers maps to a gesture name (e.g. all 5 up = Open Palm, 0 up = Fist).
- `_fingers_up()` handles the thumb separately since it bends sideways rather than up/down, and accounts for which hand (left/right) is shown to the camera.
- The gesture → action mapping in `GESTURE_ACTIONS` is a **demo stand-in**: in a full product this would trigger a real system action (e.g. media keys via `pyautogui`), but printing to console keeps the demo dependency-free and safe to run anywhere.

## Verified
The gesture-classification logic (`classify_gesture`) was unit-tested against synthetic landmark data for all 5 gestures on both hands — all pass. (See the included test approach in the project notes if you want to re-run it yourself with `pytest`.)

## Troubleshooting
- **"Could not open webcam"** — another app (Zoom, Teams, etc.) may be using the camera, or `camera_index=0` isn't your default camera; try `camera_index=1` in `run_webcam_demo()`.
- **Model download fails** — your network may be blocking `storage.googleapis.com`; download the file manually from the URL printed in the error and place it next to the script as `hand_landmarker.task`.
- **Gesture not detected / "Unknown"** — make sure your whole hand is clearly visible and well-lit; MediaPipe needs a reasonably unobstructed view.

## Submission Checklist
- [ ] Push this code to a GitHub repo named `Syntecxhub_Hand_Gesture_Recognition`
- [ ] Share internship status on LinkedIn, tagging @Syntecxhub
- [ ] Submit via the official Submission Form
