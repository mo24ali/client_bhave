import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE

from src import config
from src.utils.mlflow_utils import setup_mlflow

def compare_cluster_impact():
    # 1. Initialisation de l'expérience MLflow
    setup_mlflow(experiment_name="Telco_Cluster_Comparison")

    # 2. Chargement des données prétraitées et clusterisées
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv")

    cols_to_drop = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)
    if 'Profil_Client' in df.columns:
        cols_to_drop.append('Profil_Client')

    X_full = df.drop(columns=cols_to_drop)
    y = df[config.TARGET_COL]

    # 3. Séparation Train / Test globale et stratifiée
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # Création des deux variantes de jeux de données (Sans Cluster / Avec Cluster)
    X_train_no_cluster = X_train_full.drop(columns=['Cluster'], errors='ignore')
    X_test_no_cluster = X_test_full.drop(columns=['Cluster'], errors='ignore')

    X_train_with_cluster = X_train_full.copy()
    X_test_with_cluster = X_test_full.copy()

    # 4. Modèles à tester pour la comparaison
    models_config = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {'classifier__C': [0.1, 1, 10]}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [10, 20]}
        },
        "Decision_Tree":{
            "model": DecisionTreeClassifier(random_state=42),
            "params": {'classifier__max_depth': [5, 10, 20]}
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.1]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {'classifier__C': [0.1, 1, 10], 'classifier__kernel':['linear', 'rbf']}
        }
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trusted_types = [
        "imblearn.pipeline.Pipeline",
        "imblearn.over_sampling._smote.base.SMOTE",
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "sklearn.neighbors._kd_tree.KDTree",
        "sklearn.tree._tree.Tree"
    ]

    print("Début de la comparaison stricte (Sans Cluster vs Avec Cluster)...")

    # 5. Boucle sur les deux scénarios
    scenarios = {
        "Sans_Cluster": (X_train_no_cluster, X_test_no_cluster),
        "Avec_Cluster": (X_train_with_cluster, X_test_with_cluster)
    }

    for scenario_name, (X_tr, X_te) in scenarios.items():
        print(f"\n--- SCÉNARIO : {scenario_name} ---")

        for model_name, cfg in models_config.items():
            run_name = f"{model_name}_{scenario_name}"
            print(f"  Entraînement de {run_name}...")

            with mlflow.start_run(run_name=run_name):
                pipeline = ImPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('classifier', cfg["model"])
                ])

                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=cfg["params"],
                    n_iter=2,
                    scoring='recall',
                    cv=cv,
                    random_state=42,
                    n_jobs=-1
                )

                search.fit(X_tr, y_train.values.ravel())
                best_model = search.best_estimator_

                # Évaluation sur le test set correspondant
                y_pred = best_model.predict(X_te)
                y_proba = best_model.predict_proba(X_te)[:, 1]

                metrics = {
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba)
                }

                mlflow.log_params(search.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    best_model, 
                    f"model_{run_name}",
                    skops_trusted_types=trusted_types
                )

                print(f"    {run_name} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")

    print("\nComparaison terminée ! Vous pouvez maintenant comparer les runs dans MLflow UI.")

if __name__ == "__main__":
    compare_cluster_impact()