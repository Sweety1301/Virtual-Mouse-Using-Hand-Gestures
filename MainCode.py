# ---------------- IMPORT REQUIRED LIBRARIES ----------------

import cv2                      # OpenCV for video capture & image processing
import mediapipe as mp          # MediaPipe for hand landmark detection
import pyautogui                # Control mouse & keyboard actions
import random                   # Generate random numbers (for screenshot names)
import numpy as np              # Numerical operations (distance, angle math)
from pynput.mouse import Button, Controller  # Low-level mouse control
import time                     # Timing & cooldown management


# ---------------- CONFIGURATION CONSTANTS ----------------
# These constants improve readability and allow easy tuning

ANGLE_BENT = 50                 # Angle threshold for bent finger
ANGLE_STRAIGHT = 90             # Angle threshold for straight finger
PINCH_THRESHOLD = 0.05          # Distance threshold for pinch detection
MOVE_THRESHOLD = 0.07           # Minimum distance to allow cursor movement
CLICK_COOLDOWN = 0.3            # Delay between consecutive clicks
SCREENSHOT_COOLDOWN = 1.0       # Delay between screenshots
SMOOTHING_FACTOR = 0.2          # Cursor smoothing factor (EMA)


# ---------------- INITIAL SETUP ----------------

mouse = Controller()            # Initialize mouse controller
screen_width, screen_height = pyautogui.size()  # Get screen resolution


# ---------------- MEDIAPIPE HAND INITIALIZATION ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     # Continuous video stream
    model_complexity=1,          # Balance between accuracy & speed
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1              # Track only one hand
)


# ---------------- GLOBAL STATE VARIABLES ----------------

prev_x, prev_y = 0, 0           # Previous cursor position (for smoothing)
last_click_time = 0             # Timestamp of last click
last_screenshot_time = 0        # Timestamp of last screenshot


# ---------------- HELPER FUNCTIONS ----------------

def get_distance(p1, p2):
    """
    Calculates Euclidean distance between two normalized points.
    Used for pinch & gesture detection.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_angle(a, b, c):
    """
    Calculates angle (in degrees) at point b formed by points a-b-c.
    Used to determine finger bending.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians * 180 / np.pi)

    # Normalize angle to [0, 180]
    return 360 - angle if angle > 180 else angle


def find_finger_tip(processed):
    """
    Extracts index finger tip landmark from MediaPipe output.
    """
    if processed.multi_hand_landmarks:
        return processed.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
    return None


# ---------------- GESTURE DETECTION FUNCTIONS ----------------

def is_left_click(landmarks, dist):
    """
    Detects left click gesture:
    - Index finger bent
    - Middle finger straight
    - Thumb away from index
    """
    return (
        get_angle(landmarks[5], landmarks[6], landmarks[8]) < ANGLE_BENT and
        get_angle(landmarks[9], landmarks[10], landmarks[12]) > ANGLE_STRAIGHT and
        dist > PINCH_THRESHOLD
    )


def is_right_click(landmarks, dist):
    """
    Detects right click gesture:
    - Middle finger bent
    - Index finger straight
    - Thumb away from index
    """
    return (
        get_angle(landmarks[9], landmarks[10], landmarks[12]) < ANGLE_BENT and
        get_angle(landmarks[5], landmarks[6], landmarks[8]) > ANGLE_STRAIGHT and
        dist > PINCH_THRESHOLD
    )


def is_double_click(landmarks, dist):
    """
    Detects double click gesture:
    - Index & middle fingers bent
    - Thumb away from index
    """
    return (
        get_angle(landmarks[5], landmarks[6], landmarks[8]) < ANGLE_BENT and
        get_angle(landmarks[9], landmarks[10], landmarks[12]) < ANGLE_BENT and
        dist > PINCH_THRESHOLD
    )


def is_screenshot(landmarks, dist):
    """
    Detects screenshot gesture:
    - All fingers folded (fist)
    - Thumb pinched
    """
    fingers_folded = (
        landmarks[8][1] > landmarks[6][1] and
        landmarks[12][1] > landmarks[10][1] and
        landmarks[16][1] > landmarks[14][1] and
        landmarks[20][1] > landmarks[18][1]
    )
    return fingers_folded and dist < PINCH_THRESHOLD


# ---------------- MAIN GESTURE EXECUTION ----------------

def detect_gesture(frame, landmarks, processed):
    """
    Maps detected gestures to mouse actions.
    """
    global prev_x, prev_y, last_click_time, last_screenshot_time

    # Ensure all landmarks are present
    if len(landmarks) < 21:
        return

    # Get index finger tip position
    index_tip = find_finger_tip(processed)

    # Compute distance between thumb & index finger tip
    thumb_index_dist = get_distance(landmarks[4], landmarks[8])

    # Current time for cooldown handling
    current_time = time.time()


    # ---------- MOUSE MOVEMENT ----------
    # Move cursor only when index finger is straight & thumb is away
    if thumb_index_dist > MOVE_THRESHOLD and \
       get_angle(landmarks[5], landmarks[6], landmarks[8]) > ANGLE_STRAIGHT:

        if index_tip:
            # Map hand coordinates to screen coordinates
            target_x = int(index_tip.x * screen_width)
            target_y = int(index_tip.y * screen_height)

            # Apply smoothing to avoid jitter
            smooth_x = int(prev_x + (target_x - prev_x) * SMOOTHING_FACTOR)
            smooth_y = int(prev_y + (target_y - prev_y) * SMOOTHING_FACTOR)

            # Move mouse
            pyautogui.moveTo(smooth_x, smooth_y)

            # Update previous position
            prev_x, prev_y = smooth_x, smooth_y


    # ---------- LEFT CLICK ----------
    elif is_left_click(landmarks, thumb_index_dist) and \
         current_time - last_click_time > CLICK_COOLDOWN:

        mouse.click(Button.left)
        last_click_time = current_time

        cv2.putText(frame, "Left Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # ---------- RIGHT CLICK ----------
    elif is_right_click(landmarks, thumb_index_dist) and \
         current_time - last_click_time > CLICK_COOLDOWN:

        mouse.click(Button.right)
        last_click_time = current_time

        cv2.putText(frame, "Right Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # ---------- DOUBLE CLICK ----------
    elif is_double_click(landmarks, thumb_index_dist) and \
         current_time - last_click_time > CLICK_COOLDOWN:

        pyautogui.doubleClick()
        last_click_time = current_time

        cv2.putText(frame, "Double Click", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


    # ---------- SCREENSHOT ----------
    elif is_screenshot(landmarks, thumb_index_dist) and \
         current_time - last_screenshot_time > SCREENSHOT_COOLDOWN:

        # Save screenshot with random filename
        label = random.randint(1000, 9999)
        pyautogui.screenshot(f"my_screenshot_{label}.png")

        last_screenshot_time = current_time

        cv2.putText(frame, "Screenshot Taken", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ---------------- MAIN APPLICATION LOOP ----------------

def main():
    """
    Captures webcam feed, processes hand landmarks,
    and executes gestures in real time.
    """
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Limit camera FPS for performance
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            processed = hands.process(rgb)

            landmarks = []

            # If hand detected, extract landmarks
            if processed.multi_hand_landmarks:
                hand = processed.multi_hand_landmarks[0]

                # Draw landmarks on frame
                draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Store normalized landmark coordinates
                landmarks = [(lm.x, lm.y) for lm in hand.landmark]

                # Detect and execute gesture
                detect_gesture(frame, landmarks, processed)

            # Display output window
            cv2.imshow("Virtual Mouse Control", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------- PROGRAM ENTRY POINT ----------------

if __name__ == "__main__":
    main()
