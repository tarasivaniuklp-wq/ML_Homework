import requests

url = "http://localhost:9696/predict"

sample_house = {
    "rooms": 3,
    "type": "h",
    "distance": 10.5,
    "bathroom": 2,
    "car": 1,
    "landsize": 450,
    "buildingarea": 125,
    "yearbuilt": 1975,
    "councilarea": "Yarra",
    "lattitude": -37.8,
    "longtitude": 144.9,
    "propertycount": 4000,
}

response = requests.post(url, json=sample_house)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())