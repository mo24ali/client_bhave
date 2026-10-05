import os
import mlflow



def setup_mlflow(experiment_name="Telco_Churn_Prediction", tracking_uri="file:./mlruns"):
    os.makedirs(tracking_uri.replace("file:", "").replace("./", ""), exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"✅ MLflow configuré.")
    print(f"📁 Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"🧪 Expérience : {experiment_name}")

if __name__ == "__main__":
    setup_mlflow()