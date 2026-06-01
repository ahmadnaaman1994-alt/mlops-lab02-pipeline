import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_path = "data/train"
test_path = "data/test"

processed_train_path = "data/processed/train"
processed_test_path = "data/processed/test"

os.makedirs(processed_train_path, exist_ok=True)
os.makedirs(processed_test_path, exist_ok=True)

train_frames = []
for file in os.listdir(train_path):
    df = pd.read_csv(os.path.join(train_path, file))
    train_frames.append(df)

train_data = pd.concat(train_frames, ignore_index=True)

scaler = StandardScaler()
scaler.fit(train_data[["hour"]])

for file in os.listdir(train_path):
    df = pd.read_csv(os.path.join(train_path, file))

    X_scaled = scaler.transform(df[["hour"]])

    df_processed = pd.DataFrame({
        "hour": X_scaled.flatten(),
        "orders": df["orders"]
    })

    df_processed.to_csv(os.path.join(processed_train_path, file), index=False)

for file in os.listdir(test_path):
    df = pd.read_csv(os.path.join(test_path, file))

    X_scaled = scaler.transform(df[["hour"]])

    df_processed = pd.DataFrame({
        "hour": X_scaled.flatten(),
        "orders": df["orders"]
    })

    df_processed.to_csv(os.path.join(processed_test_path, file), index=False)

print("Preprocessing completed.")