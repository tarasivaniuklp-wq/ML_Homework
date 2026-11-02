import pickle
from sklearn.linear_model import LinearRegression

with open('pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
test = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

X = dv.transform([test])
y_pred = model.predict_proba(X)[0][1]

print(round(y_pred, 3))   