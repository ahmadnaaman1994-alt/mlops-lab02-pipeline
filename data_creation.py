import os
import numpy as np
import pandas as pd

np.random.seed(42)

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

def generate_restaurant_orders_data(n_points=24, noise_level=3.0, add_anomalies=False):
    hour = np.arange(n_points)

    lunch_peak = 25 * np.exp(-0.5 * ((hour - 13) / 2) ** 2)
    dinner_peak = 35 * np.exp(-0.5 * ((hour - 19) / 2) ** 2)
    base_orders = 8 + lunch_peak + dinner_peak

    orders = base_orders + np.random.normal(0, noise_level, n_points)
    orders = np.round(np.clip(orders, 0, None)).astype(int)

    if add_anomalies:
        anomaly_indices = np.random.choice(n_points, size=3, replace=False)
        anomaly_values = np.random.choice([20, -15, 25], size=3)
        orders[anomaly_indices] = np.clip(orders[anomaly_indices] + anomaly_values, 0, None)

    df = pd.DataFrame({
        "hour": hour,
        "orders": orders
    })

    return df

for i in range(1, 4):
    df_train = generate_restaurant_orders_data(
        n_points=24,
        noise_level=2.0,
        add_anomalies=False
    )
    df_train.to_csv(f"train/data_{i}.csv", index=False)

for i in range(1, 3):
    df_test = generate_restaurant_orders_data(
        n_points=24,
        noise_level=4.0,
        add_anomalies=True
    )
    df_test.to_csv(f"test/data_{i}.csv", index=False)

print("Creation completed: restaurant orders data saved in 'train' and 'test' folders.")