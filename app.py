import os
import streamlit as st
import pandas as pd
import joblib
from src import config

# Configuration de la page Streamlit
st.set_page_config(
    page_title="ClientBehave - Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# Chargement du modèle Champion
@st.cache_resource
def load_champion_model():
    model_path = "models_saved/churn_champion_model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# Chargement du dataset pour récupérer les colonnes et des exemples
@st.cache_data
def load_data():
    data_path = config.PROCESSED_DATA_DIR / "telco_clustered_pca.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

model = load_champion_model()
df = load_data()

st.title("📉 ClientBehave : Application de Prédiction du Churn")
st.markdown("""
Cette application interactive exploite votre **Modèle Champion** (sérialisé et optimisé via MLflow) 
pour prédire le risque de résiliation des clients de l'opérateur télécom.
""")

if model is None:
    st.error("⚠️ Aucun modèle champion trouvé dans `models_saved/churn_champion_model.pkl`. Veuillez exécuter votre script d'entraînement !")
elif df is None:
    st.error("⚠️ Fichier de données `telco_clustered_pca.csv` introuvable dans le dossier processed.")
else:
    st.success("✅ Modèle Champion et données chargés avec succès !")

    # Préparation des colonnes features attendues par le modèle
    cols_to_drop = [config.TARGET_COL]
    if config.ID_COL in df.columns:
        cols_to_drop.append(config.ID_COL)
    if 'Profil_Client' in df.columns:
        cols_to_drop.append('Profil_Client')
        
    X_features = df.drop(columns=cols_to_drop)

    # Onglets de navigation
    tab1, tab2 = st.tabs(["📊 Tester un client du Dataset", "🔍 Simuler un profil personnalisé"])

    with tab1:
        st.subheader("Sélection d'un client existant")
        st.write("Choisissez une ligne du dataset pour évaluer la prédiction du modèle face à la réalité.")

        client_index = st.number_input("Index du client (Ligne)", min_value=0, max_value=len(X_features)-1, value=0, step=1)
        
        selected_row = X_features.iloc[[client_index]]
        actual_churn = df.loc[client_index, config.TARGET_COL]

        st.markdown("**Aperçu des caractéristiques du client :**")
        st.dataframe(selected_row)

        if st.button("Lancer la prédiction pour ce client"):
            prediction = model.predict(selected_row)[0]
            probability = model.predict_proba(selected_row)[0][1]

            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error(f"🚨 **Prédiction : Risque de Churn**\n\nProbabilité de départ : **{probability * 100:.2f}%**")
                else:
                    st.success(f"✅ **Prédiction : Client Fidèle**\n\nProbabilité de départ : **{probability * 100:.2f}%**")
            
            with col_res2:
                status_reel = "A résilié (Churn)" if actual_churn == 1 else "Resté fidèle"
                st.info(f"📋 **Vérité terrain (Réalité) :**\n\n{status_reel}")

    with tab2:
        st.subheader("Simulation dynamique d'un profil client")
        st.info("Modifiez les valeurs des principales composantes ou du cluster pour tester différents scénarios.")

        with st.form("simulation_form"):
            user_inputs = {}
            # On affiche des champs de saisie pour les colonnes principales (ex: Cluster et quelques colonnes PCA)
            cols = st.columns(3)
            for i, col in enumerate(X_features.columns):
                with cols[i % 3]:
                    default_val = float(X_features[col].mean())
                    user_inputs[col] = st.number_input(f"{col}", value=default_val)

            submit_sim = st.form_submit_button("Calculer la prédiction du profil")

        if submit_sim:
            sim_df = pd.DataFrame([user_inputs])
            pred_sim = model.predict(sim_df)[0]
            proba_sim = model.predict_proba(sim_df)[0][1]

            st.divider()
            if pred_sim == 1:
                st.error(f"🚨 **Risque Élevé de Churn détecté !** (Probabilité : {proba_sim * 100:.2f}%)")
                st.markdown("💡 *Conseil commercial :* Envoyer une offre promotionnelle de fidélisation.")
            else:
                st.success(f"✅ **Client Stable (Pas de Churn)** (Probabilité de départ : {proba_sim * 100:.2f}%)")
                st.markdown("👍 *Statut :* Aucun signal d'alarme critique.")