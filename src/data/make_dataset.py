import logging
import pandas as pd
from src import config

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_raw_data() -> pd.DataFrame:
    """
        Charge le fichier csv brut depuis le chemin defini dans config.py
    """

    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f" Le fichier brut est introuvable : {config.RAW_DATA_PATH}")

    logging.info(f"Chargement des données brutes depuis : {config.RAW_DATA_PATH}")
    df = pd.read_csv(config.RAW_DATA_PATH)
    logging.info(f"Données brutes chargées avec siccès : {df.shape[0]} lignes, {df.shape[1]} colonnes .")
    return df

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(str).str.strip()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    return df_clean
def save_processed_data(df: pd.DataFrame) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    logging.info(f"fichier nettoyé sauvegardé sous : {config.CLEANED_DATA_PATH}")


def main():
    df_raw = load_raw_data()
    df_cleanned = clean_total_charges(df_raw)
    save_processed_data(df_cleanned)
    logging.info(f"Nettoyage terminé avec succès !")


if __name__ == "__main__":
    main()
    
