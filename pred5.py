import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load
from fuzzywuzzy import fuzz, process
import difflib

# Fonction pour charger et préparer les données
def load_data(filepath):
    donnees = pd.read_csv(filepath, sep=';', encoding='iso-8859-1')
    donnees['Date dépôt GED'] = pd.to_datetime(donnees['Date dépôt GED'], format='%d/%m/%Y', errors='coerce')
    donnees['Nombre d\'indices'] = donnees.groupby(['TYPE DE DOCUMENT', 'Categ_Docs','desc_lot', 'Libellé du document'])['INDICE'].transform('nunique')
    donnees = donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot', 'Nombre d\'indices']]
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_vars = encoder.fit_transform(donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot']]).toarray()
    encoded_vars_df = pd.DataFrame(encoded_vars, columns=encoder.get_feature_names_out())
    
    df_final = pd.concat([encoded_vars_df, donnees[['Nombre d\'indices']]], axis=1)
    
    X = df_final.drop(columns=['Nombre d\'indices'])
    y = df_final['Nombre d\'indices']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), encoder, donnees

# Fonction pour rechercher les meilleurs hyperparamètres et entraîner les modèles
def hyperparameter_search_and_train(X_train, y_train):
    # Régression linéaire (pas de recherche d'hyperparamètres pour ce modèle)
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Forêt Aléatoire
    rf_model = RandomForestRegressor(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    rf_grid_search.fit(X_train, y_train)
    best_rf_model = rf_grid_search.best_estimator_
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gb_grid_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
    gb_grid_search.fit(X_train, y_train)
    best_gb_model = gb_grid_search.best_estimator_
    
    return model_lr, best_rf_model, best_gb_model

# Fonction pour évaluer les modèles
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
    return results

# Charger les données et l'encodeur
(X_train, X_test, y_train, y_test), encoder, donnees = load_data('GOODLIFE_pré.csv')

# Rechercher les meilleurs hyperparamètres et entraîner les modèles
model_lr, best_rf_model, best_gb_model = hyperparameter_search_and_train(X_train, y_train)

# Évaluer les modèles
models = {
    'Régression Linéaire': model_lr,
    'Forêt Aléatoire': best_rf_model,
    'Gradient Boosting': best_gb_model
}
results = evaluate_models(models, X_test, y_test)

# Sélectionner le meilleur modèle
best_model_name = min(results, key=lambda x: results[x]['MAE'])
best_model = models[best_model_name]

# Enregistrer le meilleur modèle
dump(best_model, 'best_model.joblib')
dump(encoder, 'encoder.joblib')

# Fonction pour trouver des suggestions pour des catégories inconnues
def find_similar_category(category, category_list):
    similar = difflib.get_close_matches(category, category_list)
    return similar

# Fonction pour corriger les erreurs typographiques
def correct_typo(input_str, known_categories):
    closest_match = process.extractOne(input_str, known_categories, scorer=fuzz.token_set_ratio)
    return closest_match[0] if closest_match[1] >= 80 else input_str

# Application Streamlit
st.title('Prédiction du Nombre Moyen d\'Indices')

st.write("### Résultats des modèles")
results_df = pd.DataFrame(results).T
st.write(results_df)

st.write(f"### Le meilleur modèle est : {best_model_name}")

st.write("### Faites des prédictions")
type_doc = st.text_input('Type de Document')
categ_docs = st.text_input('Catégorie de Document')
desc_lot = st.text_input('Description du Lot')

if st.button('Prédire'):
    # Correction des erreurs typographiques
    known_type_docs = donnees['TYPE DE DOCUMENT'].unique()
    known_categ_docs = donnees['Categ_Docs'].unique()
    known_desc_lot = donnees['desc_lot'].unique()
    
    corrected_type_doc = correct_typo(type_doc, known_type_docs)
    corrected_categ_docs = correct_typo(categ_docs, known_categ_docs)
    corrected_desc_lot = correct_typo(desc_lot, known_desc_lot)
    
    input_data = pd.DataFrame([[corrected_type_doc, corrected_categ_docs, corrected_desc_lot]], columns=['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot'])
    
    suggestions = {}
    
    if corrected_type_doc not in known_type_docs:
        suggestions['TYPE DE DOCUMENT'] = find_similar_category(corrected_type_doc, known_type_docs)
    if corrected_categ_docs not in known_categ_docs:
        suggestions['Categ_Docs'] = find_similar_category(corrected_categ_docs, known_categ_docs)
    if corrected_desc_lot not in known_desc_lot:
        suggestions['desc_lot'] = find_similar_category(corrected_desc_lot, known_desc_lot)
    
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

