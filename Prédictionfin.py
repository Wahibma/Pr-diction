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
def charger_donnees(fichier_telecharge=None):
    if fichier_telecharge is not None:
        donnees = pd.read_csv(fichier_telecharge, sep=';', encoding='iso-8859-1')
    else:
        donnees = pd.read_csv('GOODLIFE_pré.csv', sep=';', encoding='iso-8859-1')
        
    donnees['Date dépôt GED'] = pd.to_datetime(donnees['Date dépôt GED'], format='%d/%m/%Y', errors='coerce')
    donnees['Nombre d\'indices'] = donnees.groupby(['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot', 'Libellé du document'])['INDICE'].transform('nunique')
    donnees = donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot', 'Nombre d\'indices']]
    
    encodeur = OneHotEncoder(handle_unknown='ignore')
    variables_encodees = encodeur.fit_transform(donnees[['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot']]).toarray()
    variables_encodees_df = pd.DataFrame(variables_encodees, columns=encodeur.get_feature_names_out())
    
    df_final = pd.concat([variables_encodees_df, donnees[['Nombre d\'indices']]], axis=1)
    
    X = df_final.drop(columns=['Nombre d\'indices'])
    y = df_final['Nombre d\'indices']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), encodeur, donnees

# Fonction pour entraîner et évaluer les modèles avec GridSearchCV
def entrainer_et_evaluer_modeles(X_train, X_test, y_train, y_test):
    resultats = {}
    modeles = {}
    
    # Régression linéaire
    modele_rl = LinearRegression()
    modele_rl.fit(X_train, y_train)
    y_pred_rl = modele_rl.predict(X_test)
    resultats['Régression Linéaire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_rl),
        'MSE': mean_squared_error(y_test, y_pred_rl),
        'R2': r2_score(y_test, y_pred_rl)
    }
    modeles['Régression Linéaire'] = (modele_rl, y_pred_rl)
    
    # Forêt Aléatoire avec GridSearchCV (réduire le nombre de paramètres)
    modele_fa = RandomForestRegressor(random_state=42)
    grille_param_fa = {
        'n_estimators': [100, 200],
        'max_features': ['auto'],
        'max_depth': [10, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    recherche_fa = GridSearchCV(estimator=modele_fa, param_grid=grille_param_fa, cv=3, n_jobs=-1, verbose=1)
    recherche_fa.fit(X_train, y_train)
    meilleur_modele_fa = recherche_fa.best_estimator_
    y_pred_fa = meilleur_modele_fa.predict(X_test)
    resultats['Forêt Aléatoire'] = {
        'MAE': mean_absolute_error(y_test, y_pred_fa),
        'MSE': mean_squared_error(y_test, y_pred_fa),
        'R2': r2_score(y_test, y_pred_fa)
    }
    modeles['Forêt Aléatoire'] = (meilleur_modele_fa, y_pred_fa)
    
    # Gradient Boosting avec GridSearchCV (réduire le nombre de paramètres)
    modele_gb = GradientBoostingRegressor(random_state=42)
    grille_param_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1],
        'max_depth': [3, 4],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    recherche_gb = GridSearchCV(estimator=modele_gb, param_grid=grille_param_gb, cv=3, n_jobs=-1, verbose=1)
    recherche_gb.fit(X_train, y_train)
    meilleur_modele_gb = recherche_gb.best_estimator_
    y_pred_gb = meilleur_modele_gb.predict(X_test)
    resultats['Gradient Boosting'] = {
        'MAE': mean_absolute_error(y_test, y_pred_gb),
        'MSE': mean_squared_error(y_test, y_pred_gb),
        'R2': r2_score(y_test, y_pred_gb)
    }
    modeles['Gradient Boosting'] = (meilleur_modele_gb, y_pred_gb)
    
    return resultats, modeles

# Fonction pour sélectionner le meilleur modèle
def selectionner_meilleur_modele(resultats):
    meilleur_modele_nom = min(resultats, key=lambda x: resultats[x]['MAE'])
    return meilleur_modele_nom

# Fonction pour trouver des suggestions pour des catégories inconnues
def trouver_categorie_similaire(categorie, liste_categories):
    similaire = difflib.get_close_matches(categorie, liste_categories)
    return similaire

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
fichier_telecharge = st.file_uploader("Choisissez un fichier CSV", type="csv")

# Charger les données et l'encodeur
(X_train, X_test, y_train, y_test), encodeur, donnees = charger_donnees(fichier_telecharge)

# Entraîner et évaluer les modèles
resultats, modeles = entrainer_et_evaluer_modeles(X_train, X_test, y_train, y_test)

# Sélectionner le meilleur modèle
meilleur_modele_nom = selectionner_meilleur_modele(resultats)
meilleur_modele, meilleur_modele_pred = modeles[meilleur_modele_nom]

# Enregistrer le meilleur modèle
dump(meilleur_modele, 'meilleur_modele.joblib')
dump(encodeur, 'encodeur.joblib')

col1, col2 = st.columns(2)

with col1:
    st.header("1. Résultats des modèles")
    st.write("Les résultats des différents modèles de machine learning utilisés pour la prédiction :")
    resultats_df = pd.DataFrame(resultats).T
    st.write(resultats_df)
    st.write(f"**Le Meilleur Modèle est : {meilleur_modele_nom}**")
    st.write("""
    ### Explication des Modèles
    - **Régression Linéaire** : Ce modèle est simple et rapide à exécuter. Il est utile pour comprendre la relation linéaire entre les variables indépendantes et la variable dépendante. Cependant, il peut manquer de précision pour des relations complexes et non linéaires.
    - **Forêt Aléatoire** : Ce modèle est un ensemble d'arbres de décision qui améliore la précision et la robustesse de la prédiction. Il est efficace pour capturer les relations non linéaires et pour gérer les données avec de nombreuses variables. Cependant, il peut être moins interprétable que la régression linéaire.
    - **Gradient Boosting** : Ce modèle construit des arbres de décision de manière séquentielle, chaque arbre corrigant les erreurs des arbres précédents. Il est très précis et efficace pour les relations complexes, mais peut être plus difficile à interpréter et plus long à entraîner.
    """)

    # Effet des modalités de chaque variable
    st.header("2. Effet des modalités de chaque variable")
    if meilleur_modele_nom in ['Forêt Aléatoire', 'Gradient Boosting']:
        importances = modeles[meilleur_modele_nom][0].feature_importances_
        noms_variables = encodeur.get_feature_names_out()
        importance_df = pd.DataFrame({'Variable': noms_variables, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        st.write(importance_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Variable', data=importance_df, ax=ax)
        ax.set_title(f'Importance des Variables pour le modèle {meilleur_modele_nom}')
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
    types_docs_connus = donnees['TYPE DE DOCUMENT'].unique()
    categories_docs_connues = donnees['Categ_Docs'].unique()
    descriptions_lot_connues = donnees['desc_lot'].unique()

    type_doc = st.selectbox('Type de Document', types_docs_connus, key='type_doc', help='Sélectionnez le type de document.')
    categ_docs = st.selectbox('Catégorie de Document', categories_docs_connues, key='categ_docs', help='Sélectionnez la catégorie de document.')
    desc_lot = st.selectbox('Description du Lot', descriptions_lot_connues, key='desc_lot', help='Sélectionnez la description du lot.')

    if st.button('Prédire'):
        # Vérification de la cohérence des données
        if type_doc not in types_docs_connus:
            st.error(f"Type de Document '{type_doc}' n'est pas reconnu.")
        elif categ_docs not in categories_docs_connues:
            st.error(f"Catégorie de Document '{categ_docs}' n'est pas reconnue.")
        elif desc_lot not in descriptions_lot_connues:
            st.error(f"Description du Lot '{desc_lot}' n'est pas reconnue.")
        else:
            # Encoder les variables d'entrée
            donnees_entree = pd.DataFrame([[type_doc, categ_docs, desc_lot]], columns=['TYPE DE DOCUMENT', 'Categ_Docs', 'desc_lot'])
            variables_encodees = encodeur.transform(donnees_entree).toarray()

            # Charger le modèle
            modele = load('meilleur_modele.joblib')

            # Prédire
            prediction = modele.predict(variables_encodees)
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
