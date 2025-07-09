import os
import pandas as pd

data_folder = "gesture_data"

for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_folder, filename)
        df = pd.read_csv(filepath, header=None)
        print(f"{filename}: {len(df)} samples")
