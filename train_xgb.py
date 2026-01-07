import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

os.environ["DATABRICKS_TOKEN"] = os.environ.get("DATABRICKS_TOKEN")
mlflow.set_experiment("/Users/sumalatha.suresh.nayak@gmail.com/xgboost/XGBOOST")

df = pd.read_csv("data/train.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.1,
    "n_estimators": 200
}

with mlflow.start_run():
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)

    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)

    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="xgboost_sales_model"
    )

    print(f"RMSE: {rmse}")
