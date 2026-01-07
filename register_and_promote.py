

from mlflow.tracking import MlflowClient

MODEL_NAME = "xgboost_sales_model"
client = MlflowClient()

versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
if not versions:
    raise Exception("No new model version found")

latest_version = versions[0].version

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"Model version {latest_version} promoted to Production")
