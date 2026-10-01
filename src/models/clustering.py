import logging
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from src import config


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    logging.info("=== STARTING CLUSTERING PIPELINE ===")

    train_path = config.PROCESSED_DATA_DIR / "telco_features.csv"
    logging.info(f"Loading preprocessed training data from {train_path}")
    df_train = pd.read_csv(train_path)


    cols_to_drop = [config.TARGET_COL, config.ID_COL]
    cols_to_drop = [col for col in cols_to_drop if col in df_train.columns]
    X_train = df_train.drop(columns=cols_to_drop)


    OPTIMAL_K = 4 # to be changeable based on the needs but based on the graphs in the notebook elbow => k=4, and silhouette => k=2 so I choosed 3
    logging.info(f"Training K-Means with k={OPTIMAL_K}")
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    kmeans.fit(X_train)


    config.MODELS_SAVED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, config.CLASSIFIER_MODEL_PATH)
    logging.info(f"K-Means modle saved successfully to {config.CLASSIFIER_MODEL_PATH}")

    df_train['Cluster'] = kmeans.labels_
    clustered_path = config.PROCESSED_DATA_DIR / "train_clustered.csv"
    df_train.to_csv(clustered_path, index=False)
    logging.info(f"Clustered training data saved to {clustered_path}")

    logging.info("==== CLUSTERING PIPELINE COMPLETED SUCCESSFULLYYYY ====")


if __name__ == "__main__":
    main()
    

