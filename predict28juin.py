import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import difflib
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour charger et préparer les données
def load_data(uploaded_file):
    donnees = pd.read_csv(uploaded_file, sep=';', encoding='iso-8859-1')
    donnees['Date dépôt GED'] = pd.to_datetime(donnees['Date dépôt GED'], format='%d/%m/%Y', errors='coerce')
    donnees['Nombre d\'indices'] = donnees.groupby(['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot', 'Libellé du document'])['INDICE'].transform('nunique')
    donnees = donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot', 'Nombre d\'indices']]
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_vars = encoder.fit_transform(donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot']]).toarray()
    encoded_vars_df = pd.DataFrame(encoded_vars, columns=encoder.get_feature_names_out())
    
    df_final = pd.concat([encoded_vars_df, donnees[['Nombre d\'indices']]], axis=1)
    
    X = df_final.drop(columns=['Nombre d\'indices'])
    y = df_final['Nombre d\'indices']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), encoder, donnees

# Fonction pour entraîner et évaluer les modèles avec GridSearchCV
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    models = {}
    
    # Régression linéaire
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    results['Régression Linéaire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    models['Régression Linéaire'] = (model_lr, y_pred_lr)
    
    # Forêt Aléatoire avec GridSearchCV (réduire le nombre de paramètres)
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['auto'],
        'max_depth': [10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=1)
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    y_pred_rf = best_rf_model.predict(X_test)
    results['Forêt Aléatoire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }
    models['Forêt Aléatoire'] = (best_rf_model, y_pred_rf)
    
    # Gradient Boosting avec GridSearchCV (réduire le nombre de paramètres)
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1],
        'max_depth': [3, 4],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=3, n_jobs=-1, verbose=1)
    gb_grid_search.fit(X_train, y_train)
    best_gb_model = gb_grid_search.best_estimator_
    y_pred_gb = best_gb_model.predict(X_test)
    results['Gradient Boosting'] = {
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'MSE': mean_squared_error(y_test, y_pred_gb),
        'R2': r2_score(y_test, y_pred_gb)
    }
    models['Gradient Boosting'] = (best_gb_model, y_pred_gb)
    
    return results, models

# Fonction pour sélectionner le meilleur modèle
def select_best_model(results):
    best_model_name = min(results, key=lambda x: results[x]['MAE'])
    return best_model_name

# Fonction pour trouver des suggestions pour des catégories inconnues
def find_similar_category(category, category_list):
    similar = difflib.get_close_matches(category, category_list)
    return similar

# Définir la configuration de la page
st.set_page_config(page_title="Prédiction du Nombre Moyen d'Indices", layout="wide")

# Titre de l'application
st.title('🔍 Outil de Prédiction du Nombre Moyen d\'Indices')

# Description de l'application
st.write("""
### Description
Cet outil de prédiction utilise des techniques de machine learning pour estimer le nombre moyen d'indices en fonction de différents types de documents, catégories de documents et descriptions de lots. 
Il s'agit d'un outil puissant et facile à utiliser pour obtenir des prédictions précises.
""")

# Téléchargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Charger les données et l'encodeur
    (X_train, X_test, y_train, y_test), encoder, donnees = load_data(uploaded_file)

    # Entraîner et évaluer les modèles
    results, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Sélectionner le meilleur modèle
    best_model_name = select_best_model(results)
    best_model, best_model_pred = models[best_model_name]

    # Enregistrer le meilleur modèle
    dump(best_model, 'best_model.joblib')
    dump(encoder, 'encoder.joblib')

    col1, col2 = st.columns(2)

    with col1:
        st.header("Résultats des Modèles")
        st.write("Les résultats des différents modèles de machine learning utilisés pour la prédiction :")
        results_df = pd.DataFrame(results).T
        st.write(results_df)
        st.write(f"**Le Meilleur Modèle est : {best_model_name}**")
        st.write("""
        ### Explication des Modèles
        - **Régression Linéaire** : Ce modèle est simple et rapide à exécuter. Il est utile pour comprendre la relation linéaire entre les variables indépendantes et la variable dépendante. Cependant, il peut manquer de précision pour des relations complexes et non linéaires.
        - **Forêt Aléatoire** : Ce modèle est un ensemble d'arbres de décision qui améliore la précision et la robustesse de la prédiction. Il est efficace pour capturer les relations non linéaires et pour gérer les données avec de nombreuses variables. Cependant, il peut être moins interprétable que la régression linéaire.
        - **Gradient Boosting** : Ce modèle construit des arbres de décision de manière séquentielle, chaque arbre corrigant les erreurs des arbres précédents. Il est très précis et efficace pour les relations complexes, mais peut être plus difficile à interpréter et plus long à entraîner.
        """)

        # Importance des variables
        st.header("Importance des Variables")
        if best_model_name in ['Forêt Aléatoire', 'Gradient Boosting']:
            importances = models[best_model_name][0].feature_importances_
            feature_names = encoder.get_feature_names_out()
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            st.write(importance_df)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title(f'Importance des Variables pour le modèle {best_model_name}')
            st.pyplot(fig)

    with col2:
        st.header("Faites des Prédictions")
        st.write("""
        ### Instructions
        1. **Type de Document** : Sélectionnez le type de document dans la liste déroulante.
        2. **Catégorie de Document** : Sélectionnez la catégorie de document dans la liste déroulante.
        3. **Description du Lot** : Sélectionnez la description du lot dans la liste déroulante.
        4. Cliquez sur le bouton **Prédire** pour obtenir la prédiction du nombre moyen d'indices.
        """)

        # Obtenir les valeurs uniques pour les listes déroulantes
        known_type_docs = donnees['TYPE DE DOCUMENT'].unique()
        known_categ_docs = donnees['Categ_Docs'].unique()
        known_desc_lot = donnees['desc_lot'].unique()

        type_doc = st.selectbox('Type de Document', known_type_docs, key='type_doc', help='Sélectionnez le type de document.')
        categ_docs = st.selectbox('Catégorie de Document', known_categ_docs, key='categ_docs', help='Sélectionnez la catégorie de document.')
        desc_lot = st.selectbox('Description du Lot', known_desc_lot, key='desc_lot', help='Sélectionnez la description du lot.')

        if st.button('Prédire'):
            # Vérification de la cohérence des données
            if type_doc not in known_type_docs:
                st.error(f"Type de Document '{type_doc}' n'est pas reconnu.")
            elif categ_docs not in known_categ_docs:
                st.error(f"Catégorie de Document '{categ_docs}' n'est pas reconnue.")
            elif desc_lot not in known_desc_lot:
                st.error(f"Description du Lot '{desc_lot}' n'est pas reconnue.")
            else:
                # Encoder les variables d'entrée
                input_data = pd.DataFrame([[type_doc, categ_docs, desc_lot]], columns=['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot'])
                encoded_vars = encoder.transform(input_data).toarray()

                # Charger le modèle
                model = load('best_model.joblib')

                # Prédire
                prediction = model.predict(encoded_vars)
                st.success(f"Nombre moyen d'indices prédit : {prediction[0]}")

# Ajouter un style personnalisé
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    .stButton>button:hover {
        background-color: #00509e;
    }
</style>
""", unsafe_allow_html=True)
