import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle
import os

# Load all CSV files from gesture_data
data_folder = "gesture_data"
all_data = []

for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_folder, filename), header=None)
        all_data.append(df)

# Combine into one dataset
full_data = pd.concat(all_data)
labels = full_data.iloc[:, 0]
features = full_data.iloc[:, 1:]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as gesture_model.pkl")
