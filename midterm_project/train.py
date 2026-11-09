import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """Loads dataset from CSV and performs cleaning, imputation, and feature engineering."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # 1. Drop high-cardinality, redundant, or leaking columns
    columns_to_drop = ["suburb", "address", "sellerg", "postcode", "date", "bedroom2"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # 2. Impute missing values
    df["car"] = df["car"].fillna(0)
    df["buildingarea"] = df.groupby("rooms")["buildingarea"].transform(
        lambda x: x.fillna(x.median())
    )
    df["buildingarea"] = df["buildingarea"].fillna(df["buildingarea"].median())

    # Convert yearbuilt to property age (dataset reference year: 2017)
    df["yearbuilt"] = df["yearbuilt"].fillna(df["yearbuilt"].median())
    df["age"] = 2017 - df["yearbuilt"]
    df = df.drop(columns=["yearbuilt"])

    df["councilarea"] = df["councilarea"].fillna("Unknown")

    return df


def train():
    """Trains the tuned XGBoost Regressor on the full dataset and saves the pipeline."""
    # Load cleaned data
    df = load_and_clean_data("data/melb_data.csv")

    # Apply log transformation to the target variable
    y = np.log1p(df["price"].values)
    df_features = df.drop(columns=["price"])

    print("Vectorizing categorical features...")
    dicts = df_features.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)

    print("Training final XGBoost model with tuned hyperparameters...")
    best_params = {
        "subsample": 0.8,
        "reg_lambda": 0.5,
        "reg_alpha": 0.1,
        "n_estimators": 500,
        "min_child_weight": 1,
        "max_depth": 7,
        "learning_rate": 0.03,
        "colsample_bytree": 0.6,
        "random_state": 1,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**best_params)
    model.fit(X, y)

    # Evaluate performance on full training set
    preds = model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, preds))
    print(f"Full Train RMSE: {train_rmse:.4f}")

    # Save artifacts
    output_file = "xgb_model.bin"
    print(f"Saving model and DictVectorizer to {output_file}...")
    with open(output_file, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print("Training pipeline successfully completed!")


if __name__ == "__main__":
    train()