import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def test_prediction():
    print("🔄 Chargement du modèle Champion...")
    model_path = "models_saved/churn_champion_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ Erreur : Aucun modèle champion trouvé dans 'models_saved/'. Exécutez d'abord l'entraînement.")
        return

    print("🔄 Chargement des données de test...")
    df = pd.read_csv(config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv")
    
    cols_to_drop = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)
    if 'Profil_Client' in df.columns:
        cols_to_drop.append('Profil_Client')
        
    X = df.drop(columns=cols_to_drop)
    y = df[config.TARGET_COL]
    
    # On isole le même jeu de test
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sélectionner 5 clients aléatoires pour le test
    sample_clients = X_test.sample(5, random_state=42)
    sample_actuals = y_test.loc[sample_clients.index]

    print("\n🔍 Lancement des prédictions sur 5 clients tests :\n")
    
    # Prédictions
    predictions = model.predict(sample_clients)
    probabilities = model.predict_proba(sample_clients)[:, 1] # Probabilité de Churn (classe 1)

    for i, (idx, row) in enumerate(sample_clients.iterrows()):
        pred = predictions[i]
        prob = probabilities[i]
        actual = sample_actuals.loc[idx]
        
        status = "🚨 Risque de Churn" if pred == 1 else "✅ Fidèle"
        print(f"Client n°{i+1} (ID index: {idx}) :")
        print(f"  - Prédiction du modèle : {status} (Probabilité : {prob * 100:.2f}%)")
        print(f"  - Réalité (Vérité terrain) : {'A résilié' if actual == 1 else 'Resté fidèle'}")
        print("-" * 50)

if __name__ == "__main__":
    test_prediction()