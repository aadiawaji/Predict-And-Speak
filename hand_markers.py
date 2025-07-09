import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def run_hand_detection():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
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

            cv2.imshow('Hand Tracker', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_detection()
