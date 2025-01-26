from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="Interface")  # Explicitly specify Interface as the template folder

# Check current directory and ensure template exists
print("Current directory:", os.getcwd())
print("Templates path exists:", os.path.exists("Interface/index.html"))

# Load the saved model
model_path = "Model/gbm_model.pkl"
with open(model_path, "rb") as file:
    gbm_model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def predict_price():
    prediction = None
    if request.method == "POST":
        try:
            # Retrieve form data
            features = [float(request.form[f"feature{i}"]) for i in range(1, 11)]

            # Predict using the model
            prediction = gbm_model.predict([features])[0]
        except Exception as e:
            print(f"Error: {e}")

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
