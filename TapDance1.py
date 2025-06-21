# with

import cv2
import mediapipe as mp
import random
import time
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Circle properties
circle_radius = 30
hand_circle_radius = 30
hitbox_scale_factor = 5
min_edge_distance = 50
# Add two more circle positions for feet (left and right)
foot_circle_radius = 30
foot_hitbox_scale_factor = 1.5 # CHANGED from 5

score = 0
start_time = 0
game_duration = 120

# Function to get a random position for the hand circles
def get_random_position(image_width, image_height):
    x = random.randint(min_edge_distance + circle_radius, image_width - min_edge_distance - circle_radius)
    y = random.randint(min_edge_distance + circle_radius, image_height - min_edge_distance - circle_radius)
    return (x, y)

# Function to get a random position for the foot circles
def get_random_foot_position(image_width, image_height):
    x = random.randint(
        min_edge_distance + foot_circle_radius,
        image_width - min_edge_distance - foot_circle_radius
    )

    # Force the foot target to be right above the bottom edge
    y = image_height - foot_circle_radius - min_edge_distance

    return (x, y)

# Initial circle positions
circle_positions = [
    get_random_position(640, 480),
    get_random_position(640, 480)
]

def is_body_part_in_circle(body_part_x, body_part_y, circle_x, circle_y, radius, hitbox_scale_factor):
    adjusted_radius = radius * hitbox_scale_factor
    distance_squared = (body_part_x - circle_x) ** 2 + (body_part_y - circle_y) ** 2
    return distance_squared <= adjusted_radius ** 2

def display_menu():
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define button rectangles: (x, y, width, height)
    button_start = (220, 220, 200, 50)
    button_leaderboard = (220, 300, 200, 50)
    button_quit = (220, 380, 200, 50)

    choice = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_start[0] <= x <= button_start[0] + button_start[2] and button_start[1] <= y <= button_start[1] + button_start[3]:
                choice = "start"
            elif button_leaderboard[0] <= x <= button_leaderboard[0] + button_leaderboard[2] and button_leaderboard[1] <= y <= button_leaderboard[1] + button_leaderboard[3]:
                choice = "leaderboard"
            elif button_quit[0] <= x <= button_quit[0] + button_quit[2] and button_quit[1] <= y <= button_quit[1] + button_quit[3]:
                choice = "quit"

    cv2.namedWindow('Menu')
    cv2.setMouseCallback('Menu', mouse_callback)

    while True:
        frame = 255 * np.ones(shape=[600, 640, 3], dtype=np.uint8)

        # Title text
        menu_text = "Dance Game"
        (w1, h1) = cv2.getTextSize(menu_text, font, 1.5, 2)[0]
        cv2.putText(frame, menu_text, (int((640 - w1) / 2), 150), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw buttons rectangles and text
        cv2.rectangle(frame, (button_start[0], button_start[1]), (button_start[0] + button_start[2], button_start[1] + button_start[3]), (0, 255, 0), -1)
        cv2.putText(frame, "Start", (button_start[0] + 70, button_start[1] + 35), font, 1, (255, 255, 255), 2)

        cv2.rectangle(frame, (button_leaderboard[0], button_leaderboard[1]), (button_leaderboard[0] + button_leaderboard[2], button_leaderboard[1] + button_leaderboard[3]), (255, 0, 0), -1)
        cv2.putText(frame, "Leaderboard", (button_leaderboard[0] + 30, button_leaderboard[1] + 35), font, 1, (255, 255, 255), 2)

        cv2.rectangle(frame, (button_quit[0], button_quit[1]), (button_quit[0] + button_quit[2], button_quit[1] + button_quit[3]), (0, 0, 255), -1)
        cv2.putText(frame, "Quit", (button_quit[0] + 80, button_quit[1] + 35), font, 1, (255, 255, 255), 2)

        cv2.imshow('Menu', frame)

        if choice is not None:
            cv2.destroyWindow('Menu')
            if choice == "quit":
                cv2.destroyAllWindows()
                exit()
            return choice

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC key also quits
            cv2.destroyAllWindows()
            exit()

def display_leaderboard():
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
    
    try:
        with open('leaderboard.txt', 'r') as file:
            entries = []
            for line in file:
                name, score = line.strip().split(': ')
                entries.append((name, int(score)))
            entries.sort(key=lambda x: x[1], reverse=True)
            lines = ["Leaderboard"] + [f"{name}: {score}" for name, score in entries]
    except FileNotFoundError:
        lines = ["Leaderboard", "No leaderboard data yet."]
    
    y_position = 40
    for line in lines:
        cv2.putText(frame, line.strip(), (50, y_position), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        y_position += 30
    
    cv2.imshow('Leaderboard', frame)
    cv2.waitKey(0)

def add_to_leaderboard(name, score):
    with open('leaderboard.txt', 'a') as file:
        file.write(f"{name}: {score}\n")

def capture_player_name(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = ""
    max_length = 20
    while True:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (40, 30), (600, 140), (220, 220, 220), -1)
        cv2.putText(frame_copy, "Enter your name:", (50, 80), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, name, (50, 120), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Enter Name", frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key in (8, 127):  # Backspace
            name = name[:-1]
        elif len(name) < max_length and 32 <= key <= 126:
            name += chr(key)

    cv2.destroyWindow("Enter Name")
    return name

def is_foot_in_circle(foot_x, foot_y, circle_x, circle_y, radius, hitbox_scale_factor):
    adjusted_radius = radius * hitbox_scale_factor
    distance_squared = (foot_x - circle_x) ** 2 + (foot_y - circle_y) ** 2
    return distance_squared <= adjusted_radius ** 2

def capture_game_duration(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    duration_str = ""
    max_length = 4
    while True:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (40, 30), (620, 140), (220, 220, 220), -1)
        cv2.putText(frame_copy, "Enter game duration (seconds):", (50, 80), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame_copy, duration_str, (50, 120), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Enter Duration", frame_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            try:
                duration = int(duration_str)
                if duration > 0:
                    break
                else:
                    duration_str = ""
            except ValueError:
                duration_str = ""
        elif key in (8, 127):
            duration_str = duration_str[:-1]
        elif len(duration_str) < max_length and 48 <= key <= 57:
            duration_str += chr(key)

    cv2.destroyWindow("Enter Duration")
    return int(duration_str)

def main():
    global score, start_time, game_duration

    choice = display_menu()

    if choice == "leaderboard":
        display_leaderboard()
        return

    blank_frame = 255 * np.ones(shape=[480, 640, 3], dtype=np.uint8)
    player_name = capture_player_name(blank_frame)
    game_duration = capture_game_duration(blank_frame)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to start video capture.")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = frame.shape  # Get actual camera resolution

    # Now that we know actual size, generate foot circle positions
    foot_circle_positions = [
        get_random_foot_position(w, h),
        get_random_foot_position(w, h)
    ]

    game_over = False

    while True:
        if not game_over:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            left_hand_in_circle = False
            right_hand_in_circle = False
            left_foot_in_circle = False
            right_foot_in_circle = False

            if left_hand_in_circle and right_hand_in_circle and left_foot_in_circle and right_foot_in_circle:
                score += 1
                circle_positions[0] = get_random_position(w, h)
                circle_positions[1] = get_random_position(w, h)
                foot_circle_positions[0] = get_random_foot_position(w, h)
                foot_circle_positions[1] = get_random_foot_position(w, h)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_foot = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]


                min_visibility = 0.6

                right_foot_visible = right_foot.visibility >= min_visibility
                left_foot_visible = left_foot.visibility >= min_visibility

                h, w, _ = image.shape
                right_hand_x, right_hand_y = int(right_hand.x * w), int(right_hand.y * h)
                left_hand_x, left_hand_y = int(left_hand.x * w), int(left_hand.y * h)
                right_foot_x, right_foot_y = int(right_foot.x * w), int(right_foot.y * h)
                left_foot_x, left_foot_y = int(left_foot.x * w), int(left_foot.y * h)

                # Draw hand circles
                cv2.circle(image, (left_hand_x, left_hand_y), hand_circle_radius, (0, 255, 0), -1)
                cv2.circle(image, (right_hand_x, right_hand_y), hand_circle_radius, (0, 0, 255), -1)

                # Draw target circles for hands (red and green)
                for i, (x, y) in enumerate(circle_positions):
                    color = (0, 255, 0) if i == 1 else (0, 0, 255)
                    cv2.circle(image, (x, y), circle_radius, color, -1)

                # Draw target circles for feet (filled blue and yellow)
                foot_colors = [(255, 0, 0), (0, 255, 255)]
                for i, (x, y) in enumerate(foot_circle_positions):
                    cv2.circle(image, (x, y), foot_circle_radius, foot_colors[i], -1)  # filled


                # Draw foot circles (landmark positions) on top
                cv2.circle(image, (left_foot_x, left_foot_y), foot_circle_radius, (255, 0, 0), -1)   # Blue for left foot landmark
                cv2.circle(image, (right_foot_x, right_foot_y), foot_circle_radius, (0, 255, 255), -1) # Yellow for right foot landmark


                # Check if hands are in their circles
                left_hand_in_circle = is_body_part_in_circle(left_hand_x, left_hand_y, circle_positions[1][0], circle_positions[1][1], circle_radius, hitbox_scale_factor)
                right_hand_in_circle = is_body_part_in_circle(right_hand_x, right_hand_y, circle_positions[0][0], circle_positions[0][1], circle_radius, hitbox_scale_factor)

                # Check if feet are in their circles
                left_foot_in_circle = False
                right_foot_in_circle = False

                if left_foot_visible:
                    left_foot_in_circle = is_foot_in_circle(left_foot_x, left_foot_y, foot_circle_positions[0][0], foot_circle_positions[0][1], foot_circle_radius, foot_hitbox_scale_factor)

                if right_foot_visible:
                    right_foot_in_circle = is_foot_in_circle(right_foot_x, right_foot_y, foot_circle_positions[1][0], foot_circle_positions[1][1], foot_circle_radius, foot_hitbox_scale_factor)

                # Score only if all four are in their respective circles
                if left_hand_in_circle and right_hand_in_circle and left_foot_in_circle and right_foot_in_circle:
                    score += 1
                    circle_positions[0] = get_random_position(w, h)
                    circle_positions[1] = get_random_position(w, h)
                    foot_circle_positions[0] = get_random_foot_position(w, h)
                    foot_circle_positions[1] = get_random_foot_position(w, h)


            elapsed_time = int(time.time() - start_time)
            remaining_time = max(0, game_duration - elapsed_time)

            cv2.rectangle(image, (10, 10), (300, 100), (255, 255, 255), -1)
            cv2.putText(image, f"Time: {remaining_time}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(image, f"Score: {score}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if remaining_time == 0 and not game_over:
                game_over = True
                cap.release()  # release here once game ends
        
        else:
            button_exit = (220, 300, 200, 50)
            button_home = (220, 380, 200, 50)

            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if button_exit[0] <= x <= button_exit[0] + button_exit[2] and button_exit[1] <= y <= button_exit[1] + button_exit[3]:
                        cv2.destroyAllWindows()
                        exit()
                    elif button_home[0] <= x <= button_home[0] + button_home[2] and button_home[1] <= y <= button_home[1] + button_home[3]:
                        cv2.destroyAllWindows()
                        main()  # Restart game

            cv2.setMouseCallback('Dance Game', mouse_callback)

            frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Game Over!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Draw Exit button
            cv2.rectangle(frame, (button_exit[0], button_exit[1]),
                                (button_exit[0] + button_exit[2], button_exit[1] + button_exit[3]),
                                (0, 0, 255), -1)
            cv2.putText(frame, "Exit", (button_exit[0] + 60, button_exit[1] + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw Home button
            cv2.rectangle(frame, (button_home[0], button_home[1]),
                                (button_home[0] + button_home[2], button_home[1] + button_home[3]),
                                (0, 255, 0), -1)
            cv2.putText(frame, "Home", (button_home[0] + 50, button_home[1] + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if not game_over:
            cv2.imshow('Dance Game', image)
        else:
            cv2.imshow('Dance Game', frame)


        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == 27:  # ESC key to quit from game over quickly
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
