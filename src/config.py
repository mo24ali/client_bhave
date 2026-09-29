"""
Configuration centrale et gestion dynamique des chemins pour le projet clientBehave.
Garantit la portabilité du code (Local, Notebooks, Streamlit, Docker).
"""

from pathlib import Path

# ==============================================================================
# 1. RÉSOLUTIONS DYNAMIQUES DES CHEMINS (PATHLIB)
# ==============================================================================

# Racine du projet (2 niveaux au-dessus de src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Répertoires de données
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Répertoires de code et artefacts
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
MLRUNS_DIR = BASE_DIR / "mlruns"
MODELS_SAVED_DIR = BASE_DIR / "models_saved"
APP_DIR = BASE_DIR / "app"

# Fichiers de données spécifiques
RAW_DATA_FILENAME = "wa-fn-usec-telco-customer-churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "telco_cleaned.csv"

# Artefacts de modèles sérialisés
SCALER_PATH = MODELS_SAVED_DIR / "scaler.pkl"
CLUSTERING_MODEL_PATH = MODELS_SAVED_DIR / "kmeans_model.pkl"
CLASSIFIER_MODEL_PATH = MODELS_SAVED_DIR / "best_classifier.pkl"
PREPROCESSING_PIPELINE_PATH = MODELS_SAVED_DIR / "preprocessing_pipeline.pkl"


# ==============================================================================
# 2. CONSTANTES MÉTIER ET MACHINE LEARNING
# ==============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colonnes identifiants et cibles
ID_COL = "customerID"
TARGET_COL = "Churn"

# Typage des colonnes du dataset Telco Customer Churn
NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Feature dynamique issue du clustering
CLUSTER_FEATURE_NAME = "cluster"


# ==============================================================================
# 3. CONFIGURATION MLFLOW TRACKING
# ==============================================================================

MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
EXPERIMENT_NAME_CLUSTERING = "Telco_Customer_Segmentation"
EXPERIMENT_NAME_CLASSIFICATION = "Telco_Churn_Prediction"


# ==============================================================================
# 4. CRÉATION AUTOMATIQUE DES DOSSIERS REQUIS
# ==============================================================================

def init_project_directories():
    """Crée automatiquement l'arborescence des dossiers s'ils n'existent pas."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        MLRUNS_DIR,
        MODELS_SAVED_DIR,
        APP_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Exécution automatique lors de l'import du module
init_project_directories()


if __name__ == "__main__":
    print("=" * 65)
    print(" [clientBehave] Configuration de l'environnement vérifiée !")
    print("=" * 65)
    print(f"• Racine du projet : {BASE_DIR}")
    print(f"• Dossier Data Brut : {RAW_DATA_DIR}")
    print(f"• Fichier CSV Brut  : {RAW_DATA_PATH}")
    print(f"• Fichier CSV Existe: {RAW_DATA_PATH.exists()}")
    print(f"• URI MLflow Local  : {MLFLOW_TRACKING_URI}")
    print("=" * 65)