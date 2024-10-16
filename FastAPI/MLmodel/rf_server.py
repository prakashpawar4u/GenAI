from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the input schema for the request using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the response schema
class IrisPrediction(BaseModel):
    predicted_class: int
    class_name: str

# Endpoint for making predictions
@app.post("/predict", response_model=IrisPrediction)
def predict(iris: IrisInput):
    # Convert input to numpy array
    input_data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return the prediction as a class number and class name
    class_names = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = int(prediction)
    class_name = class_names[predicted_class]

    return {"predicted_class": predicted_class, "class_name": class_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
