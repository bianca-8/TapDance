import cv2
import mediapipe as mp
import random
import time
import pygame

# music
pygame.mixer.init()
pygame.mixer.music.load('backgroundMusic.mp3')
pygame.mixer.music.set_volume(0.5)  # volume 0.0 to 1.0
pygame.mixer.music.play(-1)  # loop music
quack = pygame.mixer.Sound('quack.mp3')
fail = pygame.mixer.Sound('fail.mp3')

# mediapipe initialization
mpDrawing = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

# circle properties
circleRadius = 30
hitboxScaleFactor = 5
minEdgeDistance = 50
kneeCircleRadius = 30
kneeHitboxScaleFactor = 1.5

# game variables
score = 0
startTime = 0
duration = 120
MACBOOK_WIDTH = 1280
MACBOOK_HEIGHT = 720

# duck images
duckRight = cv2.imread('duckRed.png', cv2.IMREAD_UNCHANGED)  # duck for right hand
duckLeft = cv2.imread('duckGreen.png', cv2.IMREAD_UNCHANGED)  # duck for left hand
duckYellow = cv2.imread('duckYellow.png', cv2.IMREAD_UNCHANGED)
duckBlue = cv2.imread('duckBlue.png', cv2.IMREAD_UNCHANGED)

duckSize = 60
duckRight = cv2.resize(duckRight, (duckSize, duckSize))
duckLeft = cv2.resize(duckLeft, (duckSize, duckSize))
duckYellow = cv2.resize(duckYellow, (duckSize, duckSize))
duckBlue = cv2.resize(duckBlue, (duckSize, duckSize))

# fire images
redFire = cv2.imread('fireRed.png', cv2.IMREAD_UNCHANGED)
greenFire = cv2.imread('fireGreen.png', cv2.IMREAD_UNCHANGED)
yellowFire = cv2.imread('fireYellow.png', cv2.IMREAD_UNCHANGED)
blueFire = cv2.imread('fireBlue.png', cv2.IMREAD_UNCHANGED)

fireSize = 60 # same as duck size
redFire = cv2.resize(redFire, (fireSize, fireSize))
greenFire = cv2.resize(greenFire, (fireSize, fireSize))
yellowFire = cv2.resize(yellowFire, (fireSize, fireSize))
blueFire = cv2.resize(blueFire, (fireSize, fireSize))

# function to overlay an image
def overlayImage(img, overlay, pos):
    x, y = pos
    h, w = overlay.shape[0], overlay.shape[1]

    # check if image within bounds
    if y + h > img.shape[0] or x + w > img.shape[1] or x < 0 or y < 0:
        return img  # skip if out of bounds

    alphaOverlay = overlay[:, :, 3] / 255.0
    alphaBg = 1.0 - alphaOverlay

    for c in range(3):
        img[y:y+h, x:x+w, c] = (alphaOverlay * overlay[:, :, c] +
                                alphaBg * img[y:y+h, x:x+w, c])
    return img

# function to display the menu
def displayMenu():

    # button rectangles (x, y, width, height)
    buttonHandsOnly = (490, 190, 300, 80)
    buttonHandsFeet = (490, 290, 300, 80)
    buttonLeaderboard = (490, 390, 300, 80)
    buttonQuit = (490, 490, 300, 80)

    choice = None

    # function to check mouse clicks on buttons
    def mouseCallback(event, x, y, flags, param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if buttonHandsOnly[0] <= x <= buttonHandsOnly[0] + buttonHandsOnly[2] and buttonHandsOnly[1] <= y <= buttonHandsOnly[1] + buttonHandsOnly[3]:
                choice = "handsOnly"
            elif buttonHandsFeet[0] <= x <= buttonHandsFeet[0] + buttonHandsFeet[2] and buttonHandsFeet[1] <= y <= buttonHandsFeet[1] + buttonHandsFeet[3]:
                choice = "handsFeet"
            elif buttonLeaderboard[0] <= x <= buttonLeaderboard[0] + buttonLeaderboard[2] and buttonLeaderboard[1] <= y <= buttonLeaderboard[1] + buttonLeaderboard[3]:
                choice = "leaderboard"
            elif buttonQuit[0] <= x <= buttonQuit[0] + buttonQuit[2] and buttonQuit[1] <= y <= buttonQuit[1] + buttonQuit[3]:
                choice = "quit"

    cv2.namedWindow('Menu')
    cv2.resizeWindow('Menu', MACBOOK_WIDTH, MACBOOK_HEIGHT)
    cv2.setMouseCallback('Menu', mouseCallback)

    menuBg = cv2.imread('menu.jpg')
    menuBg = cv2.resize(menuBg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))

    while True:
        frame = menuBg.copy()

        cv2.imshow('Menu', frame)

        if choice is not None:
            cv2.destroyWindow('Menu')
            if choice == "quit":
                pygame.mixer.music.stop()
                cv2.destroyAllWindows()
                exit()
            return choice

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC key to quit
            pygame.mixer.music.stop()
            cv2.destroyAllWindows()
            exit()

# function to display the game over screen
def displayGameOver(score, rank):
    endBg = cv2.imread('end.jpg')
    endBg = cv2.resize(endBg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))

    buttonHome = (490, 360, 300, 80)
    buttonExit = (490, 500, 300, 80)
    choice = None

    def mouseCallback(event, x, y, flags, param):
        nonlocal choice
        if event == cv2.EVENT_LBUTTONDOWN:
            if buttonExit[0] <= x <= buttonExit[0] + buttonExit[2] and buttonExit[1] <= y <= buttonExit[1] + buttonExit[3]:
                choice = "exit"
            elif buttonHome[0] <= x <= buttonHome[0] + buttonHome[2] and buttonHome[1] <= y <= buttonHome[1] + buttonHome[3]:
                choice = "home"

    cv2.namedWindow("Game Over")
    cv2.setMouseCallback("Game Over", mouseCallback)

    fail.play()

    while True:
        frame = endBg.copy()
        cv2.putText(frame, f"Score: {score}", (490, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(frame, f"Rank: #{rank if rank else 'N/A'}", (490, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        cv2.imshow("Game Over", frame)

        if choice == "exit":
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()
            exit()
        elif choice == "home":
            cv2.destroyWindow("Game Over")
            return  # go back to main()

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()
            exit()

# function to get a random position for the hand targets
def randomPos(width, height):
    x = random.randint(minEdgeDistance + circleRadius, width - minEdgeDistance - circleRadius)
    y = random.randint(minEdgeDistance + circleRadius, height - minEdgeDistance - circleRadius)
    return (x, y)

# function to get a random position for the knee targets
def randomKneePos(width, height):
    x = random.randint(minEdgeDistance + kneeCircleRadius, width - minEdgeDistance - kneeCircleRadius)
    y = height - kneeCircleRadius - minEdgeDistance  # knee target near bottom edge
    return (x, y)

# function to check if a body part is within the target circle
def bodyInCircle(partX, partY, circleX, circleY, radius, scaleFactor):
    adjustedR = radius * scaleFactor
    distSq = (partX - circleX) ** 2 + (partY - circleY) ** 2
    return distSq <= adjustedR ** 2

# function to display the leaderboard
def displayLeaderboard():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.namedWindow('Leaderboard')
    leadBg = cv2.imread('leaderboard.jpg')
    leadBg = cv2.resize(leadBg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    buttonBack = (490, 580, 300, 80)
    backClicked = False

    # function to check mouse clicks on the back button
    def mouseCallback(event, x, y, flags, param):
        nonlocal backClicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if buttonBack[0] <= x <= buttonBack[0] + buttonBack[2] and buttonBack[1] <= y <= buttonBack[1] + buttonBack[3]:
                backClicked = True

    cv2.setMouseCallback('Leaderboard', mouseCallback)

    while True:
        frame = leadBg.copy()
        try:
            with open('leaderboard.txt', 'r') as file:
                entries = []
                for line in file:
                    # Format - Name: score, Mode: 1/2, Duration: ns
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        nameScore = parts[0].split(':')
                        if len(nameScore) == 2:
                            name = nameScore[0].strip()
                            scoreStr = nameScore[1].strip()
                            try:
                                scoreVal = int(scoreStr)
                            except ValueError:
                                continue
                            modeStr = parts[1].strip()
                            durationStr = parts[2].strip()
                            entries.append((name, scoreVal, modeStr, durationStr))
                # sort by score descending
                entries.sort(key=lambda x: x[1], reverse=True)
                entries = entries[:15]
                lines = [f"{name}: {score} | {mode} | {duration}" for name, score, mode, duration in entries]
        except FileNotFoundError:
            lines = ["Leaderboard", "No leaderboard data yet."]

        yPos = 150
        for line in lines:
            textSize, _ = cv2.getTextSize(line, font, 0.7, 2)
            xPos = (MACBOOK_WIDTH - textSize[0]) // 2
            cv2.putText(frame, line, (xPos, yPos), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            yPos += 29

        cv2.imshow('Leaderboard', frame)

        if backClicked:
            cv2.destroyWindow('Leaderboard')
            main()
            return

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyWindow('Leaderboard')
            return

# function to add to leaderboard
def addToLeaderboard(name, score, mode, duration):
    with open('leaderboard.txt', 'a') as file:
        file.write(f"{name}: {score}, Mode: {mode}, Duration: {duration}s\n")

# function to get player name
def playerName(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = ""
    maxLen = 20
    inputBg = cv2.imread('input.jpg')
    inputBg = cv2.resize(inputBg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Name")

    while True:
        frame = inputBg.copy()
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
        elif len(name) < maxLen and 32 <= key <= 126:
            name += chr(key)

    cv2.destroyWindow("Enter Name")
    return name.strip()

# function to get game duration
def gameDuration(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    durationStr = ""
    maxLen = 4
    durBg = cv2.imread('duration.jpg')
    durBg = cv2.resize(durBg, (MACBOOK_WIDTH, MACBOOK_HEIGHT))
    cv2.namedWindow("Enter Duration")

    while True:
        frame = durBg.copy()
        box_x1, box_y1 = 100, 150
        cv2.putText(frame, durationStr, (box_x1 + 10, box_y1 + 110), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Enter Duration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:
            try:
                duration = int(durationStr)
                if duration > 0:
                    break
                else:
                    durationStr = ""
            except ValueError:
                durationStr = ""
        elif key in (8, 127):
            durationStr = durationStr[:-1]
        elif len(durationStr) < maxLen and 48 <= key <= 57:
            durationStr += chr(key)

    cv2.destroyWindow("Enter Duration")
    return int(durationStr)

# function to run hands only mode
def handsOnly():
    global score, startTime, gameDuration
    score = 0

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    handCircles = [
        {"pos": randomPos(w, h), "lastHitTime": time.time()},
        {"pos": randomPos(w, h), "lastHitTime": time.time()}
    ]

    name = playerName(frame)
    duration = gameDuration(frame)

    startTime = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mpDrawing.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            rightHand = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
            leftHand = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]

            rightX, rightY = int(rightHand.x * w), int(rightHand.y * h)
            leftX, leftY = int(leftHand.x * w), int(leftHand.y * h)

            now = time.time()
            for i in range(2):
                if now - handCircles[i]["lastHitTime"] > 8:
                    handCircles[i]["pos"] = randomPos(w, h)
                    handCircles[i]["lastHitTime"] = now

           # right hand duck target
            x1, y1 = handCircles[0]["pos"][0] - duckSize // 2, handCircles[0]["pos"][1] - duckSize // 2
            image = overlayImage(image, duckRight, (x1, y1))

            # left hand duck target
            x2, y2 = handCircles[1]["pos"][0] - duckSize // 2, handCircles[1]["pos"][1] - duckSize // 2
            image = overlayImage(image, duckLeft, (x2, y2))

            # fire images for hands
            fireX = rightX - fireSize // 2
            fireY = rightY - fireSize // 2
            image = overlayImage(image, redFire, (fireX, fireY))

            fireX = leftX - fireSize // 2
            fireY = leftY - fireSize // 2
            image = overlayImage(image, greenFire, (fireX, fireY))

            # Check if hands in targets
            rightHit = bodyInCircle(rightX, rightY, handCircles[0]["pos"][0], handCircles[0]["pos"][1], circleRadius, hitboxScaleFactor)
            leftHit = bodyInCircle(leftX, leftY, handCircles[1]["pos"][0], handCircles[1]["pos"][1], circleRadius, hitboxScaleFactor)

            if rightHit and leftHit:
                handCircles[0]["pos"] = randomPos(w, h)
                handCircles[1]["pos"] = randomPos(w, h)
                handCircles[0]["lastHitTime"] = time.time()
                handCircles[1]["lastHitTime"] = time.time()
                quack.play()
                score += 1

        elapsed = int(time.time() - startTime)
        remaining = max(0, duration - elapsed)

        cv2.putText(image, f"Time Left: {remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

        cv2.imshow("Hands Only Mode", image)

        if remaining <= 0 or cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow("Hands Only Mode")

    addToLeaderboard(name, score, 1, duration)

    # rank leaderboard
    leaderboardEntries = []

    try:
        with open('leaderboard.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    nameScore = parts[0].split(':')
                    if len(nameScore) == 2:
                        name = nameScore[0].strip()
                        try:
                            scoreVal = int(nameScore[1].strip())
                            leaderboardEntries.append((name, scoreVal))
                        except ValueError:
                            continue
    except FileNotFoundError:
        pass

    # sort and determine rank
    leaderboardEntries.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, entry in enumerate(leaderboardEntries) if entry[0] == name and entry[1] == score), None)

    displayGameOver(score, rank)

# function to run hands and feet mode
def runHandsFeet():
    global score, startTime, gameDuration
    score = 0

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    handCircles = [
        {"pos": randomPos(w, h), "lastHitTime": time.time()},
        {"pos": randomPos(w, h), "lastHitTime": time.time()}
    ]

    leftKneeCircle = randomKneePos(w, h)
    rightKneeCircle = randomKneePos(w, h)
    leftKneeLastHitTime = time.time()
    rightKneeLastHitTime = time.time()

    name = playerName(frame)
    duration = gameDuration(frame)

    startTime = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mpDrawing.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            rightHand = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_WRIST]
            leftHand = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_WRIST]
            rightKnee = results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]
            leftKnee = results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]

            rightX, rightY = int(rightHand.x * w), int(rightHand.y * h)
            leftX, leftY = int(leftHand.x * w), int(leftHand.y * h)
            rightKneeX, rightKneeY = int(rightKnee.x * w), int(rightKnee.y * h)
            leftKneeX, leftKneeY = int(leftKnee.x * w), int(leftKnee.y * h)

            now = time.time()
            for i in range(2):
                if now - handCircles[i]["lastHitTime"] > 8:
                    handCircles[i]["pos"] = randomPos(w, h)
                    handCircles[i]["lastHitTime"] = now

            if now - leftKneeLastHitTime > 8:
                leftKneeCircle = randomKneePos(w, h)
                leftKneeLastHitTime = now

            if now - rightKneeLastHitTime > 8:
                rightKneeCircle = randomKneePos(w, h)
                rightKneeLastHitTime = now

            # right hand duck target
            x1, y1 = handCircles[0]["pos"][0] - duckSize // 2, handCircles[0]["pos"][1] - duckSize // 2
            image = overlayImage(image, duckRight, (x1, y1))

            # left hand duck target
            x2, y2 = handCircles[1]["pos"][0] - duckSize // 2, handCircles[1]["pos"][1] - duckSize // 2
            image = overlayImage(image, duckLeft, (x2, y2))

            # knee targets
            x, y = leftKneeCircle[0] - duckSize // 2, leftKneeCircle[1] - duckSize // 2
            image = overlayImage(image, duckYellow, (x, y))

            x, y = rightKneeCircle[0] - duckSize // 2, rightKneeCircle[1] - duckSize // 2
            image = overlayImage(image, duckBlue, (x, y))

            # fire images for knees
            fireX = rightKneeX - fireSize // 2
            fireY = rightKneeY - fireSize // 2
            image = overlayImage(image, blueFire, (fireX, fireY))

            fireX = leftKneeX - fireSize // 2
            fireY = leftKneeY - fireSize // 2
            image = overlayImage(image, yellowFire, (fireX, fireY))

            # fire images for hands
            fireX = rightX - fireSize // 2
            fireY = rightY - fireSize // 2
            image = overlayImage(image, redFire, (fireX, fireY))

            fireX = leftX - fireSize // 2
            fireY = leftY - fireSize // 2
            image = overlayImage(image, greenFire, (fireX, fireY))

            # hit for hands
            leftHandHit = bodyInCircle(leftX, leftY, handCircles[1]["pos"][0], handCircles[1]["pos"][1], circleRadius, hitboxScaleFactor)
            rightHandHit = bodyInCircle(rightX, rightY, handCircles[0]["pos"][0], handCircles[0]["pos"][1], circleRadius, hitboxScaleFactor)

            # hit for knees
            leftKneeHit = bodyInCircle(leftKneeX, leftKneeY, leftKneeCircle[0], leftKneeCircle[1], kneeCircleRadius, kneeHitboxScaleFactor)
            rightKneeHit = bodyInCircle(rightKneeX, rightKneeY, rightKneeCircle[0], rightKneeCircle[1], kneeCircleRadius, kneeHitboxScaleFactor)

            # move targets if both hands hit or both knees hit
            if (leftHandHit and rightHandHit) or (leftKneeHit and rightKneeHit):
                quack.play()
                score += 1

                # move hand targets only
                if leftHandHit and rightHandHit:
                    handCircles = [
                        {"pos": randomPos(w, h), "lastHitTime": time.time()},
                        {"pos": randomPos(w, h), "lastHitTime": time.time()}
                    ]

                # move knee targets only
                if leftKneeHit and rightKneeHit:
                    leftKneeCircle = randomKneePos(w, h)
                    rightKneeCircle = randomKneePos(w, h)
                    leftKneeLastHitTime = now
                    rightKneeLastHitTime = now

        elapsed = int(time.time() - startTime)
        remaining = max(0, duration - elapsed)

        cv2.putText(image, f"Time Left: {remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f"Score: {score}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

        cv2.imshow("Hands and Feet Mode", image)

        if remaining <= 0 or cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow("Hands and Feet Mode")

    addToLeaderboard(name, score, 2, duration)
    
    # rank leaderboard
    leaderboardEntries = []

    try:
        with open('leaderboard.txt', 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    nameScore = parts[0].split(':')
                    if len(nameScore) == 2:
                        name = nameScore[0].strip()
                        try:
                            scoreVal = int(nameScore[1].strip())
                            leaderboardEntries.append((name, scoreVal))
                        except ValueError:
                            continue
    except FileNotFoundError:
        pass

    # sort and determine rank
    leaderboardEntries.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, entry in enumerate(leaderboardEntries) if entry[0] == name and entry[1] == score), None)
    displayGameOver(score, rank)

# main function to run the game
def main():
    while True:
        choice = displayMenu()
        if choice == "handsOnly":
            handsOnly()
        elif choice == "handsFeet":
            runHandsFeet()
        elif choice == "leaderboard":
            displayLeaderboard()
        else:
            pygame.mixer.music.stop()
            break

if __name__ == "__main__":
    main()