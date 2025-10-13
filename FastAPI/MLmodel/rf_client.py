import requests

url = "http://127.0.0.1:8000/predict"

# Input data for the model (based on iris features)
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Make a POST request to the API
response = requests.post(url, json=data)

# Output the prediction
print("Response:", response.json())
