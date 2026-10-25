from flask import Flask
from flask import request
from flask import jsonify
from sklearn.linear_model import LogisticRegression
import pickle

with open("model_C=1.0.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)
    
app = Flask("Model")
@app.route("/predict", methods = ['POST'])

    
def predict():
    
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    
    churn = y_pred>0.5
    
    result = {
        "churn_prob": float(y_pred),
        "churn": bool(churn)
    }
    return jsonify(result)
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)