import cv2
import numpy as np
import mediapipe as mp
import random
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Game settings
screen_width, screen_height = 1280, 720
dot_radius = 40
min_distance = 80  # Minimum distance between hands to consider a "touch"

# Initialize dot positions
def random_dot_position():
    return (
        random.randint(100, screen_width - 100),
        random.randint(100, screen_height - 100)
    )

left_dot = random_dot_position()
right_dot = random_dot_position()

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, screen_width)
cap.set(4, screen_height)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror view
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            cx = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].x * screen_width)
            cy = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].y * screen_height)

            if label == "Left":
                left_hand = (cx, cy)
                cv2.circle(img, left_hand, 20, (0, 255, 0), -1)
                cv2.putText(img, 'L', (left_hand[0]-10, left_hand[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                right_hand = (cx, cy)
                cv2.circle(img, right_hand, 20, (0, 0, 255), -1)
                cv2.putText(img, 'R', (right_hand[0]-10, right_hand[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Draw target dots
    cv2.circle(img, left_dot, dot_radius, (0, 255, 0), -1)
    cv2.putText(img, 'L', (left_dot[0]-10, left_dot[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.circle(img, right_dot, dot_radius, (0, 0, 255), -1)
    cv2.putText(img, 'R', (right_dot[0]-10, right_dot[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Check for touch
    if left_hand and right_hand:
        if distance(left_hand, left_dot) < dot_radius and distance(right_hand, right_dot) < dot_radius:
            # Move dots to new positions, but keep them reasonably close together
            while True:
                new_left = random_dot_position()
                new_right = random_dot_position()
                if distance(new_left, new_right) < 350:  # Don't separate too far
                    left_dot = new_left
                    right_dot = new_right
                    break

    cv2.imshow("Hand Dance Game", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
