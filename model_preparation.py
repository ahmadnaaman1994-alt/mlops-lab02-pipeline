import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

processed_train_path = "data/processed/train"

all_data = []

for file in os.listdir(processed_train_path):
    file_path = os.path.join(processed_train_path, file)
    df = pd.read_csv(file_path)
    all_data.append(df)

train_data = pd.concat(all_data, ignore_index=True)

X_train = train_data[["hour"]]
y_train = train_data["orders"]

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "data/model.pkl")

print("Training completed and saved as data/model.pkl")