import cv2
import mediapipe as mp
import random
import time
import numpy as np
import pygame

# Music
pygame.mixer.init()
pygame.mixer.music.load('background_music.mp3')
pygame.mixer.music.set_volume(0.5)  # volume 0.0 to 1.0
pygame.mixer.music.play(-1)  # -1 loops the music indefinitely
quack = pygame.mixer.Sound('quack.mp3')
fail = pygame.mixer.Sound('fail.mp3')

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Circle properties
circle_radius = 30
hand_circle_radius = 30
hitbox_scale_factor = 5
min_edge_distance = 50
knee_circle_radius = 30
knee_hitbox_scale_factor = 1.5

score = 0
start_time = 0
game_duration = 120
MACBOOK_WIDTH = 1280
MACBOOK_HEIGHT = 720

# Duck images
duck_img_right = cv2.imread('duck_red.png', cv2.IMREAD_UNCHANGED)  # Duck for right hand
duck_img_left = cv2.imread('duck_green.png', cv2.IMREAD_UNCHANGED)  # Duck for left hand
duck_img_yellow = cv2.imread('duck_yellow.png', cv2.IMREAD_UNCHANGED)
duck_img_blue = cv2.imread('duck_blue.png', cv2.IMREAD_UNCHANGED)

duck_size = 60  # Adjust size as needed
duck_img_right = cv2.resize(duck_img_right, (duck_size, duck_size))
duck_img_left = cv2.resize(duck_img_left, (duck_size, duck_size))
duck_img_yellow = cv2.resize(duck_img_yellow, (duck_size, duck_size))
duck_img_blue = cv2.resize(duck_img_blue, (duck_size, duck_size))

# Fire images
red_fire_img = cv2.imread('fire_red.png', cv2.IMREAD_UNCHANGED)
green_fire_img = cv2.imread('fire_green.png', cv2.IMREAD_UNCHANGED)
yellow_fire_img = cv2.imread('fire_yellow.png', cv2.IMREAD_UNCHANGED)
blue_fire_img = cv2.imread('fire_blue.png', cv2.IMREAD_UNCHANGED)

fire_size = 60  # same as duck size or whatever you want
red_fire_img = cv2.resize(red_fire_img, (fire_size, fire_size))
green_fire_img = cv2.resize(green_fire_img, (fire_size, fire_size))
yellow_fire_img = cv2.resize(yellow_fire_img, (fire_size, fire_size))
blue_fire_img = cv2.resize(blue_fire_img, (fire_size, fire_size))

def overlay_image_alpha(img, overlay, pos):
    x, y = pos
    h, w = overlay.shape[0], overlay.shape[1]

    # Ensure image is within bounds
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return img  # Skip if out of bounds

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        img[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                alpha_background * img[y:y+h, x:x+w, c])
    return img

def display_menu():
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define button rectangles: (x, y, width, height)
    button_hands_only = (490, 190, 300, 80)
    button_hands_and_feet = (490, 290, 300, 80)
    button_leaderboard = (490, 390, 300, 80)
    button_quit = (490, 490, 300, 80)

    choice = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_hands_only[0] <= x <= button_hands_only[0] + button_hands_only[2] and button_hands_only[1] <= y <= button_hands_only[1] + button_hands_only[3]:
                choice = "hands_only"
            elif button_hands_and_feet[0] <= x <= button_hands_and_feet[0] + button_hands_and_feet[2] and button_hands_and_feet[1] <= y <= button_hands_and_feet[1] + button_hands_and_feet[3]:
                choice = "hands_and_feet"
            elif button_leaderboard[0] <= x <= button_leaderboard[0] + button_leaderboard[2] and button_leaderboard[1] <= y <= button_leaderboard[1] + button_leaderboard[3]:
                choice = "leaderboard"
            elif button_quit[0] <= x <= button_quit[0] + button_quit[2] and button_quit[1] <= y <= button_quit[1] + button_quit[3]:
                choice = "quit"

    cv2.namedWindow('Menu')
    cv2.resizeWindow('Menu', MACBOOK_WIDTH, MACBOOK_HEIGHT)
    cv2.setMouseCallback('Menu', mouse_callback)

    menu_bg = cv2.imread('menu.jpg')
    menu_bg = cv2.resize(menu_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))

    while True:
        frame = menu_bg.copy()

        cv2.imshow('Menu', frame)

        if choice is not None:
            cv2.destroyWindow('Menu')
            if choice == "quit":
                pygame.mixer.music.stop()
                cv2.destroyAllWindows()
                exit()
            return choice

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC key also quits
            pygame.mixer.music.stop()
            cv2.destroyAllWindows()
            exit()

def add_to_leaderboard(name, score):
    with open('leaderboard.txt', 'a') as file:
        file.write(f"{name}: {score}\n")

def display_game_over_screen(score, rank):
    end_bg = cv2.imread('end.jpg')
    end_bg = cv2.resize(end_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))

    button_home = (490, 360, 300, 80)
    button_exit = (490, 500, 300, 80)
    choice = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_exit[0] <= x <= button_exit[0] + button_exit[2] and button_exit[1] <= y <= button_exit[1] + button_exit[3]:
                choice = "exit"
            elif button_home[0] <= x <= button_home[0] + button_home[2] and button_home[1] <= y <= button_home[1] + button_home[3]:
                choice = "home"

    cv2.namedWindow("Game Over")
    cv2.setMouseCallback("Game Over", mouse_callback)

    fail.play()

    while True:
        frame = end_bg.copy()
        cv2.putText(frame, f"Score: {score}", (490, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(frame, f"Rank: #{rank if rank else 'N/A'}", (490, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        cv2.imshow("Game Over", frame)

        if choice == "exit":
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()
            exit()
        elif choice == "home":
            cv2.destroyWindow("Game Over")
            return  # Go back to main()

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()
            exit()

def capture_player_name(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = ""
    max_length = 20
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    input_bg = cv2.imread('input.jpg')
    input_bg = cv2.resize(input_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Name")

    while True:
        frame = input_bg.copy()
        box_x1, box_y1 = 100, 150
        box_x2, box_y2 = frame_width - 100, 250
        cv2.putText(frame, name, (box_x1 + 10, box_y1 + 110), font, 1, (0, 0, 0), 2)

        if name.strip() == "":
            cv2.putText(frame, "Name cannot be empty", (box_x1 + 10, box_y2 + 40), font, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enter Name", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and name.strip() != "":
            break
        elif key in (8, 127):
            name = name[:-1]
        elif len(name) < max_length and 32 <= key <= 126:
            name += chr(key)

    cv2.destroyWindow("Enter Name")
    return name.strip()

def capture_game_duration(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    duration_str = ""
    max_length = 4
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    dur_bg = cv2.imread('duration.jpg')
    dur_bg = cv2.resize(dur_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Duration")

    while True:
        frame = dur_bg.copy()
        box_x1, box_y1 = 100, 150
        box_x2, box_y2 = frame_width - 100, 250
        cv2.putText(frame, duration_str, (box_x1 + 10, box_y1 + 110), font, 1, (0, 0, 0), 2)
        cv2.imshow("Enter Duration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            try:
                duration = int(duration_str)
                if duration > 0:
                    break
            except ValueError:
                pass
            duration_str = ""
        elif key in (8, 127):
            duration_str = duration_str[:-1]
        elif len(duration_str) < max_length and 48 <= key <= 57:
            duration_str += chr(key)

    cv2.destroyWindow("Enter Duration")
    return int(duration_str)

def get_random_position(width, height):
    x = random.randint(min_edge_distance + circle_radius, width - min_edge_distance - circle_radius)
    y = random.randint(min_edge_distance + circle_radius, height - min_edge_distance - circle_radius)
    return (x, y)

def get_random_knee_position(width, height):
    x = random.randint(min_edge_distance + knee_circle_radius, width - min_edge_distance - knee_circle_radius)
    y = height - knee_circle_radius - min_edge_distance  # Knee target near bottom edge
    return (x, y)

def is_body_part_in_circle(part_x, part_y, circle_x, circle_y, radius, scale_factor):
    adjusted_radius = radius * scale_factor
    dist_sq = (part_x - circle_x) ** 2 + (part_y - circle_y) ** 2
    return dist_sq <= adjusted_radius ** 2

def display_leaderboard():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('Leaderboard')
    lead_bg = cv2.imread('leaderboard.jpg')
    lead_bg = cv2.resize(lead_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    button_back = (490, 580, 300, 80)
    back_clicked = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal back_clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_back[0] <= x <= button_back[0] + button_back[2] and button_back[1] <= y <= button_back[1] + button_back[3]:
                back_clicked = True

    cv2.setMouseCallback('Leaderboard', mouse_callback)

    while True:
        frame = lead_bg.copy()
        try:
            with open('leaderboard.txt', 'r') as file:
                entries = []
                for line in file:
                    # Expecting format: Name: score, Mode: X, Duration: Ys
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        name_score = parts[0].split(':')
                        if len(name_score) == 2:
                            name = name_score[0].strip()
                            score_str = name_score[1].strip()
                            try:
                                score_val = int(score_str)
                            except ValueError:
                                continue
                            mode_str = parts[1].strip()
                            duration_str = parts[2].strip()
                            entries.append((name, score_val, mode_str, duration_str))
                # Sort by score descending
                entries.sort(key=lambda x: x[1], reverse=True)
                entries = entries[:15]
                lines = [f"{name}: {score} | {mode} | {duration}" for name, score, mode, duration in entries]
        except FileNotFoundError:
            lines = ["Leaderboard", "No leaderboard data yet."]

        y_pos = 150
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, 0.7, 2)
            x_pos = (MACBOOK_WIDTH - text_size[0]) // 2
            cv2.putText(frame, line, (x_pos, y_pos), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            y_pos += 29

        cv2.imshow('Leaderboard', frame)

        if back_clicked:
            cv2.destroyWindow('Leaderboard')
            main()
            return

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyWindow('Leaderboard')
            return

def add_to_leaderboard(name, score, mode, duration):
    with open('leaderboard.txt', 'a') as file:
        file.write(f"{name}: {score}, Mode: {mode}, Duration: {duration}s\n")

def capture_player_name(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = ""
    max_len = 20
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    input_bg = cv2.imread('input.jpg')
    input_bg = cv2.resize(input_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Name")

    while True:
        frame = input_bg.copy()
        box_x1, box_y1 = 100, 150
        cv2.putText(frame, name, (box_x1 + 10, box_y1 + 110), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        if name.strip() == "":
            cv2.putText(frame, "Name cannot be empty", (box_x1 + 10, box_y1 + 150), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Enter Name", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and name.strip() != "":
            break
        elif key in (8, 127):
            name = name[:-1]
        elif len(name) < max_len and 32 <= key <= 126:
            name += chr(key)

    cv2.destroyWindow("Enter Name")
    return name.strip()

def capture_game_duration(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    duration_str = ""
    max_len = 4
    dur_bg = cv2.imread('duration.jpg')
    dur_bg = cv2.resize(dur_bg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Duration")

    while True:
        frame = dur_bg.copy()
        box_x1, box_y1 = 100, 150
        cv2.putText(frame, duration_str, (box_x1 + 10, box_y1 + 110), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Enter Duration", frame)
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
        elif len(duration_str) < max_len and 48 <= key <= 57:
            duration_str += chr(key)

    cv2.destroyWindow("Enter Duration")
    return int(duration_str)

def run_hands_only_mode():
    global score, start_time, game_duration
    score = 0

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    hand_circles = [get_random_position(w, h), get_random_position(w, h)]  # for left and right hand targets

    player_name = capture_player_name(frame)
    game_duration = capture_game_duration(frame)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

            right_x, right_y = int(right_hand.x * w), int(right_hand.y * h)
            left_x, left_y = int(left_hand.x * w), int(left_hand.y * h)

           # Right hand duck target
            x1, y1 = hand_circles[0][0] - duck_size // 2, hand_circles[0][1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_right, (x1, y1))

            # Left hand duck target
            x2, y2 = hand_circles[1][0] - duck_size // 2, hand_circles[1][1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_left, (x2, y2))

            # Draw outlined circles on actual detected hand positions
            x_fire = right_x - fire_size // 2
            y_fire = right_y - fire_size // 2
            image = overlay_image_alpha(image, red_fire_img, (x_fire, y_fire))

            # Similarly for left hand:
            x_fire = left_x - fire_size // 2
            y_fire = left_y - fire_size // 2
            image = overlay_image_alpha(image, green_fire_img, (x_fire, y_fire))

            # Check hits for hands: Right hand hits hand_circles[0], Left hand hits hand_circles[1]
            right_hit = is_body_part_in_circle(right_x, right_y, hand_circles[0][0], hand_circles[0][1], circle_radius, hitbox_scale_factor)
            left_hit = is_body_part_in_circle(left_x, left_y, hand_circles[1][0], hand_circles[1][1], circle_radius, hitbox_scale_factor)

            if right_hit and left_hit:
                quack.play()
                score += 1
                hand_circles = [get_random_position(w, h), get_random_position(w, h)]  # move circles

        elapsed = int(time.time() - start_time)
        remaining = max(0, game_duration - elapsed)

        cv2.putText(image, f"Time Left: {remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

        cv2.imshow("Hands Only Mode", image)

        if remaining <= 0 or cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow("Hands Only Mode")

    add_to_leaderboard(player_name, score, 1, game_duration)

    # Load and rank leaderboard
    leaderboard_entries = []
    leaderboard_entries = []
    try:
        with open('leaderboard.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    name_score = parts[0].split(':')
                    if len(name_score) == 2:
                        name = name_score[0].strip()
                        try:
                            score_val = int(name_score[1].strip())
                            leaderboard_entries.append((name, score_val))
                        except ValueError:
                            continue
    except FileNotFoundError:
        pass

    # Sort and determine rank
    leaderboard_entries.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, entry in enumerate(leaderboard_entries) if entry[0] == player_name and entry[1] == score), None)

    display_game_over_screen(score, rank)

def run_hands_and_feet_mode():
    global score, start_time, game_duration
    score = 0

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    hand_circles = [get_random_position(w, h), get_random_position(w, h)]
    left_knee_circle = get_random_knee_position(w, h)   # Yellow circle for left knee
    right_knee_circle = get_random_knee_position(w, h)  # Blue circle for right knee

    player_name = capture_player_name(frame)
    game_duration = capture_game_duration(frame)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

            right_x, right_y = int(right_hand.x * w), int(right_hand.y * h)
            left_x, left_y = int(left_hand.x * w), int(left_hand.y * h)
            right_knee_x, right_knee_y = int(right_knee.x * w), int(right_knee.y * h)
            left_knee_x, left_knee_y = int(left_knee.x * w), int(left_knee.y * h)

            # Right hand duck target
            x1, y1 = hand_circles[0][0] - duck_size // 2, hand_circles[0][1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_right, (x1, y1))

            # Left hand duck target
            x2, y2 = hand_circles[1][0] - duck_size // 2, hand_circles[1][1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_left, (x2, y2))

            # Draw the knee target circles:
            # Draw duck images for knee targets using overlay_image_alpha
            x, y = left_knee_circle[0] - duck_size // 2, left_knee_circle[1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_yellow, (x, y))

            x, y = right_knee_circle[0] - duck_size // 2, right_knee_circle[1] - duck_size // 2
            image = overlay_image_alpha(image, duck_img_blue, (x, y))

            # Draw player's feet
            x_fire = right_knee_x - fire_size // 2
            y_fire = right_knee_y - fire_size // 2
            image = overlay_image_alpha(image, blue_fire_img, (x_fire, y_fire))

            x_fire = left_knee_x - fire_size // 2
            y_fire = left_knee_y - fire_size // 2
            image = overlay_image_alpha(image, yellow_fire_img, (x_fire, y_fire))

            # Draw hand targets
            x_fire = right_x - fire_size // 2
            y_fire = right_y - fire_size // 2
            image = overlay_image_alpha(image, red_fire_img, (x_fire, y_fire))

            # Similarly for left hand:
            x_fire = left_x - fire_size // 2
            y_fire = left_y - fire_size // 2
            image = overlay_image_alpha(image, green_fire_img, (x_fire, y_fire))

            # Check hits for hands
            left_hand_hit = is_body_part_in_circle(left_x, left_y, hand_circles[1][0], hand_circles[1][1], circle_radius, hitbox_scale_factor)
            right_hand_hit = is_body_part_in_circle(right_x, right_y, hand_circles[0][0], hand_circles[0][1], circle_radius, hitbox_scale_factor)

            # Check hits for knees
            left_knee_hit = is_body_part_in_circle(left_knee_x, left_knee_y, left_knee_circle[0], left_knee_circle[1], knee_circle_radius, knee_hitbox_scale_factor)
            right_knee_hit = is_body_part_in_circle(right_knee_x, right_knee_y, right_knee_circle[0], right_knee_circle[1], knee_circle_radius, knee_hitbox_scale_factor)

            # Score and move targets if both hands hit OR both knees hit
            if (left_hand_hit and right_hand_hit) or (left_knee_hit and right_knee_hit):
                quack.play()
                score += 1

                if left_hand_hit and right_hand_hit:
                    # Move hand targets only
                    hand_circles = [get_random_position(w, h), get_random_position(w, h)]

                if left_knee_hit and right_knee_hit:
                    # Move knee targets only
                    left_knee_circle = get_random_knee_position(w, h)   # Yellow circle for left knee
                    right_knee_circle = get_random_knee_position(w, h)  # Blue circle for right knee

        elapsed = int(time.time() - start_time)
        remaining = max(0, game_duration - elapsed)

        cv2.putText(image, f"Time Left: {remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

        cv2.imshow("Hands and Feet Mode", image)

        if remaining <= 0 or cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow("Hands and Feet Mode")

    add_to_leaderboard(player_name, score, 2, game_duration)
    
    # Load and rank leaderboard
    leaderboard_entries = []
    leaderboard_entries = []
    try:
        with open('leaderboard.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    name_score = parts[0].split(':')
                    if len(name_score) == 2:
                        name = name_score[0].strip()
                        try:
                            score_val = int(name_score[1].strip())
                            leaderboard_entries.append((name, score_val))
                        except ValueError:
                            continue
    except FileNotFoundError:
        pass

    # Sort and determine rank
    leaderboard_entries.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, entry in enumerate(leaderboard_entries) if entry[0] == player_name and entry[1] == score), None)
    display_game_over_screen(score, rank)

def main():
    while True:
        choice = display_menu()
        if choice == "hands_only":
            run_hands_only_mode()
        elif choice == "hands_and_feet":
            run_hands_and_feet_mode()
        elif choice == "leaderboard":
            display_leaderboard()
        else:
            pygame.mixer.music.stop()
            break

if __name__ == "__main__":
    main()