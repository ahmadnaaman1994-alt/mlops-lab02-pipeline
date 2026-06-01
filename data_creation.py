import os
import numpy as np
import pandas as pd


DEFAULT_POINTS = 24
LUNCH_HOUR = 13
DINNER_HOUR = 19

LUNCH_SCALE = 25
DINNER_SCALE = 35

BASE_ORDERS = 8

ANOMALY_VALUES = [20, -15, 25]
ANOMALY_COUNT = 3

np.random.seed(42)

def create_directories():
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

def generate_restaurant_orders_data(n_points=24, noise_level=3.0, add_anomalies=False):
    hour = np.arange(n_points)

    lunch_peak = LUNCH_SCALE * np.exp(-0.5 * ((hour - LUNCH_HOUR) / 2) ** 2)
    dinner_peak = DINNER_SCALE * np.exp(-0.5 * ((hour - DINNER_HOUR) / 2) ** 2)
    base_orders = BASE_ORDERS + lunch_peak + dinner_peak

    orders = base_orders + np.random.normal(0, noise_level, n_points)
    orders = np.round(np.clip(orders, 0, None)).astype(int)

    if add_anomalies:
        anomaly_indices = np.random.choice(n_points, size=ANOMALY_COUNT, replace=False)
        anomaly_values = np.random.choice(ANOMALY_VALUES, size=ANOMALY_COUNT)
        orders[anomaly_indices] = np.clip(orders[anomaly_indices] + anomaly_values, 0, None)

    df = pd.DataFrame({
        "hour": hour,
        "orders": orders
    })

    return df

def main():
    create_directories()

    for i in range(1, 4):
        df_train = generate_restaurant_orders_data(
            n_points=DEFAULT_POINTS,
            noise_level=2.0,
            add_anomalies=False
        )
        df_train.to_csv(f"data/train/data_{i}.csv", index=False)

    for i in range(1, 3):
        df_test = generate_restaurant_orders_data(
            n_points=DEFAULT_POINTS,
            noise_level=4.0,
            add_anomalies=True
        )
        df_test.to_csv(f"data/test/data_{i}.csv", index=False)

    print("Creation completed: restaurant orders data saved in 'data/train' and 'data/test' folders.")

if __name__ == "__main__":
    main()