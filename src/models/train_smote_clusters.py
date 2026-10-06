import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score


from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE


from src import config
from src.utils.mlflow_utils import setup_mlflow

def load_and_prep_scenarios():
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv")

    cols_to_drop = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)
    if 'Profil_Client' in df.columns:
        cols_to_drop.append('Profil_Client')

    X_full = df.drop(columns=cols_to_drop)
    y = df[config.TARGET_COL]

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_no_cluster = X_train_full.drop(columns=['Cluster'])
    X_test_no_cluster = X_test_full.drop(columns=['Cluster'])


    X_train_with_cluster = X_train_full.copy()
    X_test_with_cluster = X_test_full.copy()
    
    return (X_train_no_cluster, X_test_no_cluster, 
            X_train_with_cluster, X_test_with_cluster, 
            y_train, y_test)

def run_smote_comparison():
    setup_mlflow(experiment_name="Telco_Churn_Prediction")

    (X_train_no, X_test_no, X_train_with, X_test_with, y_train, y_test) = load_and_prep_scenarios()

    mlflow.sklearn.autolog()


    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    scenarios = {
        "Sans_Cluster": (X_train_no, X_test_no),
        "Avec_Cluster": (X_train_with, X_test_with)
    }


    for model_name, model in models.items():
        for scenario_name, (X_train, X_test) in scenarios.items():
            run_name = f"{model_name}_{scenario_name}_SMOTE"
            print(f"\n Lancement du run : {run_name}")

            with mlflow.start_run(run_name=run_name):
                pipeline = ImPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', model)
                ])

                pipeline.fit(X_train, y_train.values.ravel())

                y_pred = pipeline.predict(X_test)
                y_proba = pipeline.predict_proba(X_test)[:, 1]

                metrics = {
                    "custom_f1_score": f1_score(y_test, y_pred),
                    "custom_recall": recall_score(y_test, y_pred),
                    "custom_roc_auc": roc_auc_score(y_test, y_proba)
                }

                mlflow.log_metrics(metrics)
                print(f"Terminé | Recall : {metrics['custom_recall']:.4f} | F1 : {metrics['custom_f1_score']:.4f}")

if __name__ == "__main__":
    run_smote_comparison()