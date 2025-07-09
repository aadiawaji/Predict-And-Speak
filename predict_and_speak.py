import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Text-to-speech engine
engine = pyttsx3.init()

# Speak asynchronously in a new thread
def speak_async(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

# Mediapipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Helper to extract (x, y) from landmarks
def extract_landmarks(landmarks):
    return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y)]

last_prediction = None
speak_cooldown = 90  # frames to wait before speaking again
cooldown_counter = 0

# Webcam loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_landmarks(hand_landmarks)
            features = np.array(features).reshape(1, -1)

            try:
                prediction = model.predict(features)[0]
                cv2.putText(image, f"Prediction: {prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Speak if different and cooldown expired
                if prediction != last_prediction and cooldown_counter <= 0:
                    speak_async(prediction)
                    last_prediction = prediction
                    cooldown_counter = speak_cooldown
                else:
                    cooldown_counter -= 1
            except:
                pass  # If model can't predict, just skip

    cv2.imshow("Sign to Speech", image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows() 