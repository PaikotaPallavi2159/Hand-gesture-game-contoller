import cv2
import mediapipe as mp
import pyautogui

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tip landmarks for fingers
tip_ids = [8, 12, 16, 20]  # Index to pinky

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    action = "NO HAND"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm_list = []
        for id, lm in enumerate(hand.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append((cx, cy))

        fingers_up = 0
        for i in tip_ids:
            # tip.y < pip.y means finger is up
            if lm_list[i][1] < lm_list[i - 2][1] - 10:  # threshold to reduce false positives
                fingers_up += 1

        # Detect open palm, closed fist, or neutral
        if fingers_up >= 4:
            pyautogui.keyDown("right")
            pyautogui.keyUp("left")
            action = "GAS (RIGHT)"
        elif fingers_up == 0:
            pyautogui.keyDown("left")
            pyautogui.keyUp("right")
            action = "BRAKE (LEFT)"
        else:
            pyautogui.keyUp("left")
            pyautogui.keyUp("right")
            action = "NEUTRAL"

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    else:
        pyautogui.keyUp("left")
        pyautogui.keyUp("right")

    # Display Action
    cv2.putText(frame, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 255, 0) if action == "GAS (RIGHT)" else (0, 0, 255) if action == "BRAKE (LEFT)" else (255, 255, 0), 4)

    cv2.imshow("Hill Climb Palm Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
