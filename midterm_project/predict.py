import pickle as pkl
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

with open("xgb_model.bin", "rb") as file:
    dv, model = pkl.load(file)
    
app = Flask("melbourne_housing_prediction")
CORS(app)

def preprocess_features(house: dict) -> dict:
    """Preprocesses input feature dictionary for single prediction."""
    house = house.copy()
    
    # Normalize keys to lowercase/snake_case if needed
    house = {k.lower().replace(" ", "_"): v for k, v in house.items()}
    
    # Convert yearbuilt to property age if yearbuilt is provided
    if "yearbuilt" in house and "age" not in house:
        house["age"] = 2017 - house["yearbuilt"]
        del house["yearbuilt"]

    # Fill defaults if missing in request payload
    house.setdefault("car", 0)
    house.setdefault("councilarea", "Unknown")

    return house

@app.route("/predict", methods=["POST"])

def predict():
    house_data = request.get_json() 
    
    if not house_data:
        return jsonify({"error": "No input JSON data provided"}), 400

    processed_house = preprocess_features(house_data)
    
    X = dv.transform([processed_house])
    
    y_pred_log = model.predict(X)[0]
    y_pred = float(np.expm1(y_pred_log))
    
    return jsonify(
        {
            "predicted_price_usd": round(y_pred, 2),
            "log_price": float(round(y_pred_log, 4)),
        }
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
    
    