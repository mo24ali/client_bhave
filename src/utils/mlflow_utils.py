import os
import mlflow

# Changement majeur ici : on utilise sqlite:///mlflow.db au lieu de file:./mlruns
def setup_mlflow(experiment_name="Telco_Churn_Prediction", tracking_uri="sqlite:///mlflow.db"):
    """
    Configure le tracking local MLflow avec une base de données SQLite.
    """
    # MLflow créera automatiquement le fichier mlflow.db à la racine de votre projet
    
    # Configuration de l'URI et de l'expérience
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow configuré avec succès.")
    print(f"📁 Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"🧪 Expérience : {experiment_name}")

if __name__ == "__main__":
    setup_mlflow()