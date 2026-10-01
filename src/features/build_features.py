import pandas as pd
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src import config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_cleaned_data() -> pd.DataFrame:
    if not config.CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"cleaned data not found at : {config.CLEANED_DATA_PATH}")

    logging.info(f"Loading cleaned data from {config.CLEANED_DATA_PATH}")
    return pd.read_csv(config.CLEANED_DATA_PATH)

def build_preprocessor() -> ColumnTransformer:
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = ["gender", "Contract", "PaymentMethod"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols), # standarization
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols) # encoding
        ],
        remainder="drop"
    )

    preprocessor.set_output(transform="pandas")

    return preprocessor

def main():
    logging.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")

    df = load_cleaned_data()

    X = df.drop(columns=[config.TARGET_COL, config.ID_COL])

    y = df[config.TARGET_COL].map({"Yes": 1, "No": 0})

    preprocessor = build_preprocessor()
    logging.info("Fitting preprocessor and transforming features . . .")
    X_processed = preprocessor.fit_transform(X)

    config.PREPROCESSING_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.PREPROCESSING_PIPELINE_PATH)
    logging.info(f"Preprocessor saved successfully to {config.PREPROCESSING_PIPELINE_PATH}")
    df_final = pd.concat([df[[config.ID_COL]], X_processed, y], axis=1)


    final_data_path = config.PROCESSED_DATA_DIR / "telco_features.csv"
    df_final.to_csv(final_data_path, index=False)
    logging.info(f"Final ML-ready dataset saved to {final_data_path}")
    logging.info("=== FEATURE ENGINEERING COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
   