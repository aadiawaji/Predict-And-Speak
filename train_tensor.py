import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# --- Load CSV data ---
data_dir = "gesture_data"
all_data = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path, header=None)
        all_data.append(df)

# Combine all gesture data
full_data = pd.concat(all_data)
labels = full_data.iloc[:, 0].values  # gesture names
features = full_data.iloc[:, 1:].values  # 42 float values (x, y) pairs

# --- Encode gesture labels to integers ---
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save label map for later (e.g., in mobile app)
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", label_map)

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# --- Normalize features (optional but helpful) ---
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# --- Build TensorFlow model ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Train model ---
model.fit(X_train, y_train, epochs=30, validation_split=0.1)

# --- Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test accuracy: {acc:.2f}")

# --- Save the model ---
model.save("gesture_model_tf.h5")
print("✅ Model saved as gesture_model_tf.h5")

# Save the label map to file
with open("label_map.txt", "w") as f:
    for label, code in label_map.items():
        f.write(f"{code},{label}\n")
