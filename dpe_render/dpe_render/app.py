"""
Application Streamlit — Estimation de la consommation énergétique d'un logement (Sujet A)

Lancement depuis ce dossier :
    streamlit run app.py
Puis ouvrir http://127.0.0.1:8501
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CHEMIN_MODELE = os.path.join(DOSSIER, "models", "modele_conso_dpe.joblib")

THERMIQUES = ["besoin_chauffage", "besoin_ecs", "deperditions_enveloppe", "deperditions_murs",
              "deperditions_baies_vitrees", "deperditions_renouvellement_air"]

LIBELLES = {
    "type_batiment": "Type de logement", "periode_construction": "Période de construction",
    "surface_habitable_logement": "Surface habitable (m²)", "hauteur_sous_plafond": "Hauteur sous plafond (m)",
    "nombre_niveau_logement": "Nombre de niveaux", "zone_climatique": "Zone climatique",
    "classe_altitude": "Altitude", "qualite_isolation_enveloppe": "Isolation de l'enveloppe",
    "qualite_isolation_murs": "Isolation des murs", "qualite_isolation_plancher_bas": "Isolation du plancher bas",
    "qualite_isolation_menuiseries": "Isolation des menuiseries", "classe_inertie_batiment": "Inertie du bâti",
    "indicateur_confort_ete": "Confort d'été", "type_energie_principale_chauffage": "Énergie de chauffage",
    "type_generateur_chauffage_principal": "Générateur de chauffage",
    "type_installation_chauffage": "Installation de chauffage", "type_energie_principale_ecs": "Énergie de l'eau chaude",
    "ubat_w_par_m2_k": "Ubat (W/m²·K)", "besoin_chauffage": "Besoin de chauffage (kWh/an)",
    "besoin_ecs": "Besoin d'eau chaude (kWh/an)", "deperditions_enveloppe": "Déperditions de l'enveloppe (W/K)",
    "deperditions_murs": "Déperditions des murs (W/K)", "deperditions_baies_vitrees": "Déperditions des vitrages (W/K)",
    "deperditions_renouvellement_air": "Déperditions par renouvellement d'air (W/K)",
}
GROUPES = {
    "Logement": ["type_batiment", "periode_construction", "surface_habitable_logement",
                 "hauteur_sous_plafond", "nombre_niveau_logement", "zone_climatique", "classe_altitude"],
    "Isolation": ["qualite_isolation_enveloppe", "qualite_isolation_murs", "qualite_isolation_plancher_bas",
                  "qualite_isolation_menuiseries", "classe_inertie_batiment", "indicateur_confort_ete"],
    "Équipements": ["type_energie_principale_chauffage", "type_generateur_chauffage_principal",
                    "type_installation_chauffage", "type_energie_principale_ecs"],
    "Relevés thermiques": ["ubat_w_par_m2_k"] + THERMIQUES,
}

st.set_page_config(page_title="Estimation de consommation", layout="wide")


# ------------------------------------------------------------------
# Chargement du modèle sérialisé (une seule fois grâce au cache)
# ------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    return joblib.load(CHEMIN_MODELE)


@st.cache_resource
def charger_explainer(_modele):
    return shap.TreeExplainer(_modele)


if not os.path.exists(CHEMIN_MODELE):
    st.error(f"Modèle introuvable : {CHEMIN_MODELE}. Lancez d'abord `python sauvegarder_modele.py`.")
    st.stop()

artefact = charger_modele()
pipeline = artefact["pipeline"]
preparation = pipeline.named_steps["preparation"]
explainer = charger_explainer(pipeline.named_steps["modele"])


def preparer(brut: pd.DataFrame) -> pd.DataFrame:
    """Même transformation que le nettoyage du notebook : grandeurs thermiques ramenées au m²."""
    X = brut.copy()
    for c in THERMIQUES:
        X[c + "_par_m2"] = X[c] / X["surface_habitable_logement"]
    return X[artefact["colonnes"]]


# ------------------------------------------------------------------
# Interface : formulaire de saisie
# ------------------------------------------------------------------
st.title("Estimation de la consommation énergétique d'un logement")
st.caption("Consommation annuelle d'énergie primaire, 5 usages, en kWh/m²/an — à partir des relevés de la visite technique")

saisie = {}
with st.form("formulaire"):
    for titre, colonnes in GROUPES.items():
        st.subheader(titre)
        cols = st.columns(3)
        for i, c in enumerate(colonnes):
            zone = cols[i % 3]
            if c in artefact["defauts_num"]:
                defaut = artefact["defauts_num"][c]
                if c == "nombre_niveau_logement":
                    saisie[c] = zone.number_input(LIBELLES[c], min_value=1, value=int(defaut), step=1)
                else:
                    saisie[c] = zone.number_input(LIBELLES[c], min_value=0.0, value=float(round(defaut, 2)))
            else:
                options = artefact["modalites"][c]
                saisie[c] = zone.selectbox(LIBELLES[c], options, index=options.index(artefact["defauts_cat"][c]))
    valider = st.form_submit_button("Prédire la consommation", type="primary")

# ------------------------------------------------------------------
# Prédiction + explication locale SHAP
# ------------------------------------------------------------------
if valider:
    # "non renseigné" = valeur manquante : la pipeline l'impute comme pendant l'entraînement
    brut = pd.DataFrame([{k: (np.nan if v == "non renseigné" else v) for k, v in saisie.items()}])
    if saisie["type_batiment"] == "maison":
        brut["type_installation_chauffage"] = "individuel"
    X = preparer(brut)
    prediction = float(pipeline.predict(X)[0])

    st.metric("Consommation estimée", f"{prediction:.0f} kWh/m²/an",
              help=f"Erreur moyenne du modèle sur le jeu de test : ± {artefact['mae_test']:.1f} kWh/m²/an")

    st.subheader("Pourquoi cette prédiction ?")
    X_prep = pd.DataFrame(preparation.transform(X), columns=artefact["noms_affichage"])
    explication = explainer(X_prep)
    # Afficher les valeurs saisies plutôt que les valeurs standardisées
    affichage = []
    for nom_brut, valeur in zip(preparation.get_feature_names_out(), X_prep.iloc[0]):
        if nom_brut in X.columns:
            v = X.iloc[0][nom_brut]
            affichage.append(f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else str(v))
        else:
            affichage.append("oui" if valeur == 1 else "non")
    explication.display_data = np.array([affichage])

    fig = plt.figure()
    shap.plots.waterfall(explication[0], max_display=12, show=False)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Lecture : en partant de la consommation moyenne du modèle E[f(x)], chaque barre rouge augmente "
               "l'estimation et chaque barre bleue la diminue, selon les caractéristiques saisies.")
