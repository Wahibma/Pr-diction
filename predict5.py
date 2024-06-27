import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
import difflib
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour charger et préparer les données
def load_data(filepath):
    donnees = pd.read_csv(filepath, sep=';', encoding='iso-8859-1')
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

# Fonction pour entraîner et évaluer les modèles
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    
    # Régression linéaire
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    results['Régression Linéaire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    
    # Forêt Aléatoire
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Forêt Aléatoire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    results['Gradient Boosting'] = {
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'MSE': mean_squared_error(y_test, y_pred_gb),
        'R2': r2_score(y_test, y_pred_gb)
    }
    
    return results, model_lr, rf_model, gb_model

# Fonction pour sélectionner le meilleur modèle
def select_best_model(results):
    best_model_name = min(results, key=lambda x: results[x]['MAE'])
    return best_model_name

# Fonction pour trouver des suggestions pour des catégories inconnues
def find_similar_category(category, category_list):
    similar = difflib.get_close_matches(category, category_list)
    return similar

# Charger les données et l'encodeur
(X_train, X_test, y_train, y_test), encoder, donnees = load_data('GOODLIFE_pré.csv')

# Entraîner et évaluer les modèles
results, model_lr, rf_model, gb_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Sélectionner le meilleur modèle
best_model_name = select_best_model(results)
best_model = {'Régression Linéaire': model_lr, 'Forêt Aléatoire': rf_model, 'Gradient Boosting': gb_model}[best_model_name]

# Enregistrer le meilleur modèle
dump(best_model, 'best_model.joblib')
dump(encoder, 'encoder.joblib')

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

# Structurer la mise en page avec des colonnes
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

    type_doc = st.selectbox('Type de Document', known_type_docs, help='Sélectionnez le type de document.')
    categ_docs = st.selectbox('Catégorie de Document', known_categ_docs, help='Sélectionnez la catégorie de document.')
    desc_lot = st.selectbox('Description du Lot', known_desc_lot, help='Sélectionnez la description du lot.')

    if st.button('Prédire'):
        # Encoder les variables d'entrée
        input_data = pd.DataFrame([[type_doc, categ_docs, desc_lot]], columns=['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot'])

        suggestions = {}

        if type_doc not in known_type_docs:
            suggestions['TYPE DE DOCUMENT'] = find_similar_category(type_doc, known_type_docs)
        if categ_docs not in known_categ_docs:
            suggestions['Categ_Docs'] = find_similar_category(categ_docs, known_categ_docs)
        if desc_lot not in known_desc_lot:
            suggestions['desc_lot'] = find_similar_category(desc_lot, known_desc_lot)

        if suggestions:
            st.error("Catégories inconnues détectées. Voici des suggestions :")
            for key, value in suggestions.items():
                st.write(f"{key} : {value}")
        else:
            encoded_vars = encoder.transform(input_data).toarray()

            # Charger le modèle
            model = load('best_model.joblib')

            # Prédire
            prediction = model.predict(encoded_vars)
            st.success(f"Nombre moyen d'indices prédit : {prediction[0]}")
            
            # Visualiser les résultats
            st.header("Visualisation des Résultats")
            fig, ax = plt.subplots()
            sns.barplot(x=['Prediction'], y=[prediction[0]], ax=ax)
            ax.set_title("Nombre Moyen d'Indices Prédit")
            st.pyplot(fig)

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
