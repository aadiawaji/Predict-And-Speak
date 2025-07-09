import tensorflow as tf

# Load the saved .h5 model
model = tf.keras.models.load_model("gesture_model_tf.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite file
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted and saved as gesture_model.tflite")
