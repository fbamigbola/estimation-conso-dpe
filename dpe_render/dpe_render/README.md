# Estimation de la consommation énergétique — Application en ligne (Render)

Application Streamlit qui prédit la consommation énergétique d'un logement (kWh/m²/an, 5 usages, énergie primaire)
à partir des relevés de la visite technique, avec une explication SHAP de chaque prédiction.
Modèle : XGBoost optimisé (MAE test 18,0 kWh/m²/an, R² 0,92).

## Contenu

| Fichier | Rôle |
|---|---|
| `app.py` | Application Streamlit |
| `models/modele_conso_dpe.joblib` | Modèle entraîné (pipeline complète) |
| `requirements.txt` | Dépendances, versions fixées (le modèle ne se recharge qu'avec les mêmes versions) |
| `.python-version` | Version de Python utilisée par Render (3.12) |
| `render.yaml` | Configuration du service Render (commandes de build et de lancement) |
| `.streamlit/config.toml` | Configuration de Streamlit pour un serveur |

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déploiement Render

- **Build command** : `pip install -r requirements.txt`
- **Start command** : `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
