from fastapi import FastAPI
from pydantic import BaseModel

from data_creation import generate_restaurant_orders_data

app = FastAPI()


class RequestData(BaseModel):
    n_points: int = 24
    noise_level: float = 3.0
    add_anomalies: bool = False


@app.get("/")
def root():
    return {"message": "Restaurant ML API is running"}


@app.post("/generate")
def generate(data: RequestData):
    df = generate_restaurant_orders_data(
        n_points=data.n_points,
        noise_level=data.noise_level,
        add_anomalies=data.add_anomalies
    )

    return df.to_dict()