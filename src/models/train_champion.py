import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImPipeline
from imblearn.over_sampling import SMOTE

from src import config
from src.utils.mlflow_utils import setup_mlflow
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def optimize_and_save_champion():
    setup_mlflow(experiment_name="Telco_churn_Prediction")

    df = pd.read_csv(config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv")

    cols_to_drop = [config.TARGET_COL]

    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)
    if 'Profil_Client' in df.columns:
        cols_to_drop.append('Profil_Client')

    X = df.drop(columns=cols_to_drop)
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Dictionnaire complet des modèles et de leurs espaces de recherche
    models_config = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                'classifier__C': [0.01, 0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'classifier__max_depth': [None, 5, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 8]
            }
        }
    }

    best_global_score = -1
    best_global_model = None
    best_global_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(" Début de l'optimisation multi-modèles (RandomizedSearchCV)...")

    # 2. Boucle d'entraînement et d'évaluation pour chaque modèle
    for model_name, cfg in models_config.items():
        print(f"\n🔍 Analyse en cours pour : {model_name}")

        with mlflow.start_run(run_name=f"{model_name}_Optimized"):
            pipeline = ImPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', cfg["model"])
            ])

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=cfg["params"],
                n_iter=5,
                scoring='recall',
                cv=cv,
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y_train.values.ravel())
            print(f"Meilleurs paramètres pour {model_name} : {search.best_params_}")

            best_model = search.best_estimator_

            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            metrics = {
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred)
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(search.best_params_)
            # mlflow.sklearn.log_model(best_model, f"model_{model_name}")
            mlflow.sklearn.log_model(
                best_model, 
                f"model_{model_name}",
                skops_trusted_types=[
                    "imblearn.pipeline.Pipeline", 
                    "imblearn.over_sampling._smote.base.SMOTE",
                    "sklearn.metrics._dist_metrics.EuclideanDistance64",
                    "sklearn.neighbors._kd_tree.KDTree",
                    "sklearn.tree._tree.Tree"
                ]
            )
            print(f" {model_name} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")

            # 3. Élection dynamique du champion global (basé sur le Recall)
            if metrics['recall'] > best_global_score:
                best_global_score = metrics['recall']
                best_global_model = best_model
                best_global_name = model_name

    # 4. Sérialisation du meilleur modèle absolu (correction du dossier en models_saved)
    os.makedirs("models_saved", exist_ok=True)
    model_path = "models_saved/churn_champion_model.pkl"
    joblib.dump(best_global_model, model_path)
    
    print(f"\n CHAMPION GLOBAL ÉLU : {best_global_name} avec un Recall de {best_global_score:.4f} !")
    print(f" Modèle sérialisé et sauvegardé localement : {model_path}")

if __name__ == "__main__":
    optimize_and_save_champion()