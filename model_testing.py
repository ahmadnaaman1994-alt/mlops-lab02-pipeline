import os
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

processed_test_path = "processed/test"

model = joblib.load("model.pkl")

all_test_data = []

for file in os.listdir(processed_test_path):
    file_path = os.path.join(processed_test_path, file)
    df = pd.read_csv(file_path)
    all_test_data.append(df)

test_data = pd.concat(all_test_data, ignore_index=True)

X_test = test_data[["hour"]]
y_test = test_data["orders"]

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("testing completed.")
print(f"MAE: {mae:.4f}")