import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from src.utils.mlflow_utils import setup_mlflow


from src import config
def load_and_prep_data():
    """
    Charge les données et s'assure que la variable 'cluster' est exclue.
    (À adapter selon les chemins réels de tes données processed)
    """
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv")


    cols_to_drop = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)

    X = df.drop(columns=cols_to_drop)
    y = df[config.TARGET_COL]

    if 'Cluster' in X.columns:
        X = X.drop(columns=['Cluster'])
    if 'Profil_Client' in X.columns:
        X = X.drop(columns=['Profil_Client'])

    X_train, X_test, y_train , y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train , y_test

def train_baselines():
    setup_mlflow(experiment_name="Telco_Churn_Prediction")

    X_train , X_test, y_train , y_test = load_and_prep_data()

    mlflow.sklearn.autolog()


    # classification algorithms 
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=12),
        "Decision_Tree": DecisionTreeClassifier(random_state=12),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=12)
    }

    for model_name, model in models.items():
        print(f"\n lancement de run pour : {model_name}")

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train.values.ravel())

            y_pred = model.fit(X_train, y_train.values.ravel())

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "custom_f1_score": f1_score(y_test, y_pred),
                "custom_recall": recall_score(y_test, y_pred),
                "custom_roc_auc": roc_auc_score(y_test, y_proba)
            }

            mlflow.log_metrics(metrics)

            print(f"✅ {model_name} terminé | F1: {metrics['custom_f1_score']:.4f} | Recall: {metrics['custom_recall']:.4f} | ROC-AUC: {metrics['custom_roc_auc']:.4f}")


if __name__ == "__main__":
    train_baselines()