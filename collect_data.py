import cv2
import mediapipe as mp
import csv
import os

# Create output folder
os.makedirs("gesture_data", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Change this to the label you're collecting
GESTURE_LABEL = "hello"   # e.g., "hello", "yes", "no"

def extract_landmarks(hand_landmarks):
    return [coord 
            for lm in hand_landmarks.landmark 
            for coord in (lm.x, lm.y)]

cap = cv2.VideoCapture(0)
data_file = open(f'gesture_data/{GESTURE_LABEL}.csv', mode='a', newline='')
csv_writer = csv.writer(data_file)

print(f"Collecting data for gesture: '{GESTURE_LABEL}'. Press 's' to save frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Press 's' to save landmark data
            key = cv2.waitKey(1)
            if key == ord('s'):
                features = extract_landmarks(hand_landmarks)
                csv_writer.writerow([GESTURE_LABEL] + features)
                print("Saved sample.")

    cv2.imshow("Collecting Gesture", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data_file.close()
cap.release()
cv2.destroyAllWindows()
