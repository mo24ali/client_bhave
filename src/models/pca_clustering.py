import pandas as pd
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src import config
import logging
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("=== STARTING PCA + CLUSTERING PIPELINE ===")


    train_path = config.PROCESSED_DATA_DIR / "telco_features.csv"
    logging.info(f"Loading preprocessed data from {train_path}")
    df_train = pd.read_csv(train_path)


    cols_to_drop = [config.TARGET_COL, config.ID_COL]
    cols_to_drop = [col for col in cols_to_drop if col in df_train.columns]
    X_train = df_train.drop(columns=cols_to_drop)


    N_COMPONENTS = 12
    logging.info(f"Applying PCA with {N_COMPONENTS} components...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_train)

    variance_conserved = pca.explained_variance_ratio_.sum()
    logging.info(f"PCA complete. Variance conserved: {variance_conserved}")
    logging.info(f"Data shape reduced from {X_train.shape[1]} to {X_pca.shape[1]} features ")
    OPTIMAL_K = 4
    logging.info(f"Training K-Means with K = {OPTIMAL_K} on PCA-reduced data ...")

    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    config.MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)

    kmeans_path = config.MODELS_SAVED_DIR / "kmeans_pca_model.pkl"
    joblib.dump(kmeans, kmeans_path)

    pca_path = config.MODELS_SAVED_DIR / "pca_transformer.pkl"
    joblib.dump(pca, pca_path)

    logging.info(f"Models saved successfully to {config.MODELS_SAVED_DIR}")

    df_train['Cluster'] = clusters
    clustered_path  =config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv"
    df_train.to_csv(clustered_path, index=False)
    logging.info(f"Clustered data saved to {clustered_path}")
logging.info("=== PCA + CLUSTERING PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()