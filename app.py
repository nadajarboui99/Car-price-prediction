import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder="Interface")

# Charger le modèle, les encodeurs, et le scaler
with open("Model/gbm_model.pkl", "rb") as file:
    gbm_model = pickle.load(file)

with open("Model/encoders.pkl", "rb") as file:
    label_encoders = pickle.load(file)

with open("Model/scaler.pkl", "rb") as file:
    scaler_standard = pickle.load(file)

# Charger les colonnes utilisées pour One-Hot Encoding
with open("Model/one_hot_columns.pkl", "rb") as file:
    one_hot_columns = pickle.load(file)

def preprocess_input(data):
    """
    Prépare les données saisies par l'utilisateur pour la prédiction.
    """
    # Supprimer les espaces des chaînes de caractères pour les colonnes catégoriques
    for col in ['Catégorie', 'Boite vitesse', 'Marque', 'Modèle', 'Énergie']:
        data[col] = data[col].strip()

    # Encodage des colonnes catégoriques (Label Encoding pour Catégorie et Boite vitesse)
    encoded_data = []
    for col in ['Catégorie', 'Boite vitesse']:
        encoder = label_encoders[col]
        encoded_value = encoder.transform([data[col]])[0]
        encoded_data.append(encoded_value)

    # Charger les colonnes utilisées pour le One-Hot Encoding
    with open("Model/one_hot_columns.pkl", "rb") as file:
        one_hot_columns = pickle.load(file)

    # Créer un DataFrame pour One-Hot Encoding
    one_hot_data = pd.DataFrame(columns=one_hot_columns)
    one_hot_data.loc[0] = 0  # Initialiser toutes les colonnes à 0

    # Mettre à jour les colonnes correspondantes pour One-Hot Encoding
    for col in ['Marque', 'Modèle', 'Énergie']:
        column_name = f"{col}_{data[col]}"
        if column_name in one_hot_columns:
            one_hot_data.at[0, column_name] = 1
        else:
            print(f"Valeur inconnue pour {col}: {data[col]}")  # Alerte ou journalisation

    # Créer la liste des colonnes numériques dans l'ordre attendu
    numeric_data = np.array([
        data['Année'], 
        data['Kilométrage'], 
        data['Puissance fiscale'], 
        data['Puissance (ch.din)'], 
        data['Cylindrée']
    ]).reshape(1, -1)

    # Standardisation des données numériques
    numeric_data = scaler_standard.transform(numeric_data)

    # Combiner les données encodées et standardisées avec le One-Hot Encoding
    final_features = np.concatenate((numeric_data.flatten(), encoded_data, one_hot_data.values.flatten())).reshape(1, -1)

    return final_features


# Route Flask
@app.route("/", methods=["GET", "POST"])
def predict_price():
    prediction = None
    if request.method == "POST":
        try:
            # Récupération des données saisies par l'utilisateur
            user_data = {
                'Année': float(request.form['Année']),
                'Kilométrage': int(request.form['Kilométrage']),
                'Puissance fiscale': int(request.form['Puissance fiscale']),
                'Puissance (ch.din)': int(request.form['Puissance (ch.din)']),
                'Cylindrée': int(request.form['Cylindrée']),
                'Boite vitesse': request.form['Boite vitesse'],
                'Catégorie': request.form['Catégorie'],
                'Marque': request.form['Marque'],
                'Modèle': request.form['Modèle'],
                'Énergie': request.form['Énergie'],
            }

            # Prétraitement des données
            features = preprocess_input(user_data)

            # Prédiction avec le modèle
            prediction = gbm_model.predict(features)[0]
        except Exception as e:
            print(f"Erreur : {e}")

    return render_template("index.html", prediction=prediction)
