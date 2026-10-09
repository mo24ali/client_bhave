# clientBehave — Segmentation & Prédiction du Churn (Telco)

Projet Data Science complet sur le jeu de données **IBM Telco Customer Churn** : analyse exploratoire, segmentation client non supervisée (**K-Means**), modèles de classification supervisée avec gestion du déséquilibre des classes (**SMOTE**), suivi d'expériences (**MLflow**) et application web interactive (**Streamlit**).

---

## 1. Sommaire

- [Objectif](#2-objectif)
- [Jeu de données](#3-jeu-de-données)
- [Architecture du projet](#4-architecture-du-projet)
- [Installation](#5-installation)
- [Pipeline d'exécution](#6-pipeline-dexécution)
- [Application Streamlit](#7-application-streamlit)
- [Suivi des expériences avec MLflow](#8-suivi-des-expériences-avec-mlflow)
- [Notebooks](#9-notebooks)
- [Résultats principaux](#10-résultats-principaux)
- [Docker](#11-docker)
- [Auteur](#12-auteur)

---

## 2. Objectif

1. **Segmenter** la clientèle de l'opérateur télécom sans utiliser la variable cible (approche non supervisée).
2. **Prédire le churn** (résiliation) avec plusieurs algorithmes de classification et comparer leurs performances.
3. **Mesurer l'apport réel de la segmentation** en comparant les modèles *sans* et *avec* la variable `Cluster`.
4. **Tracer** hyperparamètres, métriques et modèles avec MLflow.
5. **Exposer** le meilleur modèle (champion) via une application Streamlit interactive.

---

## 3. Jeu de données

| Élément | Valeur |
|---|---|
| Fichier | `data/raw/wa-fn-usec-telco-customer-churn.csv` |
| Volume | 7 043 clients × 21 colonnes |
| Cible | `Churn` (`Yes` / `No`) → déséquilibre **73,5 % / 26,5 %** |
| Identifiant | `customerID` (exclu des features) |
| Variables | 3 numériques (`tenure`, `MonthlyCharges`, `TotalCharges`) + 16 catégorielles |

**Nettoyage** : les 11 valeurs vides de `TotalCharges` (clients `tenure = 0`) sont converties puis imputées à `0.0` (`src/data/make_dataset.py`).

---

## 4. Architecture du projet

```
clientBehave/
├── app.py                          # Application Streamlit (point d'entrée web)
├── Dockerfile                      # Conteneurisation (cf. section Docker)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # Données brutes (CSV)
│   └── processed/                  # Données nettoyées / transformées / clusterisées
│
├── notebooks/
│   ├── 01_eda_cleaning.ipynb       # Analyse exploratoire & nettoyage
│   ├── 02_clustering_analysis.ipynb# Elbow / Silhouette / comparaison algos / PCA
│   └── 03_class_models_evaluation.ipynb
│
├── models_saved/                   # Artefacts sérialisés (joblib)
│   ├── churn_champion_model.pkl    # Pipeline SMOTE + classifieur champion
│   ├── preprocessing_pipeline.pkl  # ColumnTransformer (scaler + encoder)
│   ├── kmeans_model.pkl / kmeans_pca_model.pkl
│   ├── pca_transformer.pkl
│   └── best_classifier.pkl
│
├── mlruns/                         # Run artifacts (artefacts MLflow)
├── mlflow.db                       # Backend de tracking MLflow (SQLite)
│
└── src/
    ├── config.py                   # Chemins, constantes métier & ML (pathlib)
    ├── data/make_dataset.py        # Chargement + nettoyage du CSV brut
    ├── features/build_features.py  # Standardisation + One-Hot Encoding
    ├── models/
    │   ├── clustering.py           # K-Means direct sur les features
    │   ├── pca_clustering.py       # PCA (12 comps) + K-Means
    │   ├── train_baselines.py      # Baselines + autolog MLflow
    │   ├── train_smote_clusters.py # SMOTE : comparaison avec/sans cluster
    │   ├── train_custer_comparaison.py # Grid : 5 modèles × 2 scénarios
    │   ├── train_champion.py       # RandomizedSearchCV → élection du champion
    │   └── test_prediction.py      # Contrôle du modèle sur 5 clients de test
    └── utils/
        ├── mlflow_utils.py         # setup_mlflow()
        └── dashboards/             # Capture de l'UI MLflow
```

---

## 5. Installation

```bash
# 1. Cloner le dépôt
git clone git@github.com:mo24ali/client_bhave.git
cd client_bhave

# 2. Créer un environnement virtuel (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier la configuration (chemins & données)
python -m src.config
```

> Tous les scripts s'exécutent **depuis la racine du projet** sous forme de modules (`python -m src....`) afin que `src.config` soit correctement importé.

---

## 6. Pipeline d'exécution

Ordre recommandé pour reproduire l'ensemble des artefacts :

```bash
# Étape 1 — Nettoyage des données brutes
python -m src.data.make_dataset

# Étape 2 — Feature engineering (StandardScaler + One-Hot Encoding)
python -m src.features.build_features

# Étape 3 — Segmentation non supervisée (Elbow/Silhouette étudiés dans le notebook 02)
python -m src.models.clustering         # K-Means direct
python -m src.models.pca_clustering     # PCA + K-Means → telco_clustered_pca.csv

# Étape 4 — Entraînement des modèles (tracking MLflow)
python -m src.models.train_baselines             # Baselines sans rééquilibrage
python -m src.models.train_smote_clusters        # SMOTE : avec / sans cluster
python -m src.models.train_custer_comparaison    # 5 modèles × 2 scénarios (avec GridSearch)
python -m src.models.train_champion              # RandomizedSearchCV → champion

# Étape 5 — Contrôle du modèle champion sur 5 clients de test
python -m src.models.test_prediction
```

**Points clés de l'implémentation**

- Le SMOTE est appliqué **à l'intérieur d'une `imblearn.pipeline.Pipeline`**, donc uniquement sur les plis d'entraînement de la validation croisée → **aucun data leakage** par rééchantillonnage.
- La variable cible `Churn` et l'identifiant `customerID` sont **exclus des données d'entrée du clustering** (contrainte vérifiée : `n_features_in_ = 12` sur les modèles KMeans/PCA sauvegardés).
- La validation croisée est stratifiée : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

---

## 7. Application Streamlit

```bash
streamlit run app.py
# → http://localhost:8501
```

L'application charge `models_saved/churn_champion_model.pkl` et permet :

- **Onglet 1 — Client existant** : sélection d'une ligne du jeu de données, affichage des caractéristiques (dont le `Cluster`), prédiction, probabilité de churn et vérité terrain.
- **Onglet 2 — Simulation** : saisie libre des variables d'un profil pour obtenir la prédiction et une recommandation commerciale.

En cas de modèle ou de données manquants, l'application affiche un message d'erreur explicite plutôt que de planter.

---

## 8. Suivi des expériences avec MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

| Expérience (MLflow) | Contenu |
|---|---|
| `Telco_Churn_Prediction` | Baselines + runs SMOTE (autolog : matrice de confusion, courbes ROC/PR) |
| `Telco_churn_Prediction` | Optimisation `RandomizedSearchCV` des 5 modèles → élection du champion |
| `Telco_Cluster_Comparison` | Comparaison stricte **Sans cluster** vs **Avec cluster** |

Ce qui est suivi par run : **hyperparamètres** (`search.best_params_`), **métriques** (`recall`, `f1_score`, `roc_auc`) et **modèle final** (flavor `sklearn`/`skops`, rechargeable via `mlflow.sklearn.load_model("runs:/<run_id>/model_<nom>")`).

Capture de l'interface : `src/utils/dashboards/image.png`.

---

## 9. Notebooks

| Notebook | Rôle | Statut |
|---|---|---|
| `01_eda_cleaning.ipynb` | Chargement, contrôle qualité et exploration des données | ⚠️ squelette (à compléter) |
| `02_clustering_analysis.ipynb` | Courbe du coude, silhouette, comparaison KMeans / Agglomerative / DBSCAN (Silhouette, Davies-Bouldin, Calinski-Harabasz), projection PCA 2D, libellés métier des segments | ✅ complet |
| `03_class_models_evaluation.ipynb` | Évaluation des classifieurs (Precision, Recall, F1, ROC-AUC, matrices de confusion) | ⚠️ fichier vide (à créer) |

---

## 10. Résultats principaux

### 10.1 Segments clients (K-Means, k = 4)

| Cluster | Libellé métier |
|---|---|
| 0 | Abonnés récents à haut risque |
| 1 | Clients VIP fidèles |
| 2 | Profils économes (ADSL / sans internet) |
| 3 | Clients standards (vente mou) |

### 10.2 Modèle champion

**`models_saved/churn_champion_model.pkl`** — pipeline `SMOTE → SVC(linear, C=0.1)`, élu sur le **recall** (objectif métier : rattraper un maximum de clients en phase de départ).

### 10.3 Comparaison « sans cluster » vs « avec cluster »

Test set stratifié 20 %, scoring de sélection = `recall`.

| Modèle | F1 sans cluster | F1 avec cluster | Δ |
|---|---|---|---|
| Logistic Regression | 0.6108 | 0.6135 | +0.003 |
| Decision Tree | 0.6016 | 0.6036 | +0.002 |
| Random Forest | 0.6262 | 0.6239 | −0.002 |
| Gradient Boosting | 0.6152 | 0.6125 | −0.003 |
| SVM | 0.5737 | 0.5737 | 0.000 |

**Lecture** : la variable `Cluster` n'apporte pas de signal incrémental mesurable au-delà des features déjà explicatives — la segmentation reste utile à des fins **marketing / description de clientèle**, mais pas comme feature prédictive du churn. Ce résultat est traceable dans l'expérience MLflow `Telco_Cluster_Comparison`.

---

## 11. Docker

Le conteneurisation cible l'exécution de l'application Streamlit sur le port **8501** :

```bash
docker build -t clientbehave .
docker run -p 8501:8501 clientbehave
# → http://localhost:8501
```

> ⚠️ **Statut** : le `Dockerfile` à la racine est actuellement **vide (non implémenté)**, et `.dockerignore` exclut `data/processed/` ce qui empêcherait le chargement des données dans le conteneur. Ces deux points sont à corriger avant livraison.

---

## 12. Auteur

Projet réalisé dans le cadre d'un audit / dev Data Science — dépôt GitHub : [`mo24ali/client_bhave`](https://github.com/mo24ali/client_bhave).
