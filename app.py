from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle sauvegardé
model_path = "Model/gbm_model.pkl"
with open(model_path, "rb") as file:
    gbm_model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def predict_price():
    prediction = None
    if request.method == "POST":
        try:
            # Récupération des données saisies dans le formulaire
            feature1 = float(request.form["feature1"])
            feature2 = float(request.form["feature2"])
            feature3 = float(request.form["feature3"])
            feature4 = float(request.form["feature4"])
            feature5 = float(request.form["feature5"])
            feature6 = float(request.form["feature6"])
            feature7 = float(request.form["feature7"])
            feature8 = float(request.form["feature8"])
            feature9 = float(request.form["feature9"])
            feature10 = float(request.form["feature10"])

            
            # Ajouter les caractéristiques dans un tableau pour la prédiction
            features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9 , feature10]])
            
            # Prédiction avec le modèle chargé
            prediction = gbm_model.predict(features)[0]
        except Exception as e:
            print(f"Erreur : {e}")

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
