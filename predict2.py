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

# Fonction pour trouver des suggestions pour des catégories inconnues
def find_similar_category(category, category_list):
    similar = difflib.get_close_matches(category, category_list)
    return similar

# Application Streamlit
st.title('Outil de Prédiction du Nombre Moyen d\'Indices')
st.write("""
### Description
Cet outil de prédiction utilise des techniques de machine learning pour estimer le nombre moyen d'indices en fonction de différents types de documents, catégories de documents et descriptions de lots. Il s'agit d'un outil puissant et facile à utiliser pour obtenir des prédictions précises.

### Instructions
1. **Type de Document** : Sélectionnez le type de document dans la liste déroulante.
2. **Catégorie de Document** : Sélectionnez la catégorie de document dans la liste déroulante.
3. **Description du Lot** : Sélectionnez la description du lot dans la liste déroulante.
4. Cliquez sur le bouton **Prédire** pour obtenir la prédiction du nombre moyen d'indices.
""")

st.write("### Résultats des Modèles")
results_df = pd.DataFrame(results).T
st.write(results_df)

st.write(f"### Le Meilleur Modèle est : {best_model_name}")

st.write("### Faites des Prédictions")

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
        st.write("Catégories inconnues détectées. Voici des suggestions :")
        for key, value in suggestions.items():
            st.write(f"{key} : {value}")
    else:
        encoded_vars = encoder.transform(input_data).toarray()
        
        # Charger le modèle
        model = load('best_model.joblib')
        
        # Prédire
        prediction = model.predict(encoded_vars)
        st.write(f"Nombre moyen d'indices prédit : {prediction[0]}")
