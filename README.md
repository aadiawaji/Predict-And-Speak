# Predict-And-Speak
Detect trained sign language and speak it
# Sign to Speech (VS Code Edition)

This project is a Python-based Sign Language to Speech converter built using [MediaPipe](https://github.com/google/mediapipe) for hand tracking and [TensorFlow](https://www.tensorflow.org/) for gesture classification. The app collects hand landmark data, trains a custom model, and uses text-to-speech to vocalize recognized gestures.

## Features
- Collect gesture data using webcam and MediaPipe
- Train a gesture classification model using TensorFlow
- Save/export model in `.h5` and `.tflite` formats
- Real-time prediction with speech output using `pyttsx3`

## Folder Structure
