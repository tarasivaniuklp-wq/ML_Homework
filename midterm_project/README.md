# Melbourne Housing Price Prediction

Midterm Project for the Machine Learning Zoomcamp course.

## Problem Description

This project builds a predictive machine learning pipeline to estimate housing prices in Melbourne based on property features such as location coordinates, room count, land size, building area, and property age. Accurate price estimates help real estate platforms, buyers, and sellers gauge fair market values.

The dataset used is the **Melbourne Housing Snapshot** (`melb_data.csv`).

### Highlights & ML Pipeline
- **Data Cleaning & Imputation:** Dropped high-cardinality/leaking columns (`suburb`, `address`, `sellerg`, `postcode`, `date`, `bedroom2`). Imputed missing values for `car` and `buildingarea` (grouped room median), and transformed `yearbuilt` into property `age`.
- **Target Transformation:** Applied log transformation (`np.log1p`) to normalise the target `price`.
- **Model Selection & Tuning:** Evaluated Linear Regression, Decision Trees, Random Forests, and tuned XGBoost Regressor via `RandomizedSearchCV`.
- **Final Model Performance:** The tuned XGBoost Regressor achieved a validation RMSE of ~0.1818 and a final test RMSE of ~0.2224.
- **Production Stack:** Python 3.14, Pipenv, Docker containerization, Flask REST API with Gunicorn (CORS enabled), deployed live on Render.com.

---

## Repository Structure

```text
├── data/
│   └── melb_data.csv          # Dataset
├── notebook.ipynb             # Exploratory Data Analysis, Feature Engineering & Model Tuning
├── train.py                   # Autonomous training pipeline script (outputs xgb_model.bin)
├── predict.py                 # Flask REST API prediction service (with CORS)
├── test_predict.py            # Test script for API evaluation
├── index.html                 # Web Interface for interactive price prediction
├── xgb_model.bin              # Serialized DictVectorizer and XGBoost model artifacts
├── Pipfile                    # Pipenv dependencies specification
├── Pipfile.lock               # Exact dependency lockfile (Python 3.14)
├── Dockerfile                 # Production Docker configuration (Python 3.14)
└── README.md                  # Project documentation
```

---

## Live Services & Web UI

- **Public REST API Endpoint:** `https://melbourne-housing-predict.onrender.com/predict`
- **Interactive Web Interface:** Simply open `index.html` in your browser (or via GitHub Pages).

> **Note on Free Instance:** Render spins down free web services after 15 minutes of inactivity. The first request after sleep may experience a 30–50 second delay (Cold Start) while the container wakes up.

### Testing via cURL

```bash
curl -X POST https://melbourne-housing-predict.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "propertycount": 4000
  }'
```

---

## Local Development & Setup

### 1. Install Dependencies
```bash
pip install pipenv
pipenv install
pipenv shell
```

### 2. Retrain the Model
```bash
python train.py
```

### 3. Run Flask Web Service Locally
```bash
python predict.py
```

---

## Docker Containerization

### Build Docker Image
```bash
docker build -t melbourne-housing-predict .
```

### Run Container Locally
```bash
docker run -it --rm -p 9696:9696 melbourne-housing-predict
```