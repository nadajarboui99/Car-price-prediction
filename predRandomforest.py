import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Chargement des données
df = pd.read_csv('data_cleaned.csv')

# Initialiser un dictionnaire pour stocker les encodeurs de chaque colonne
label_encoders = {}

# Identifier les colonnes catégoriques
categorical_columns = df.select_dtypes(include=['object']).columns

# Encoder toutes les colonnes catégoriques
for col in categorical_columns:
    le = LabelEncoder()
    # Encoder la colonne avec les valeurs existantes
    df[col] = le.fit_transform(df[col].fillna("Unknown"))
    # Stocker l'encodeur pour référence future
    label_encoders[col] = le

# Séparer les données après l'encodage  # Supposons que "Puissance (ch.din)" est la colonne cible pour filtrer
df_missing = df[df["Puissance (ch.din)"] == 0]
df_train = df[df["Puissance (ch.din)"] != 0]

# Afficher un aperçu des datasets
print("Dataset d'entraînement :")
print(df_train.head())

print("\nDataset avec valeurs manquantes :")
print(df_missing.head())


# 3. Sélection des features pertinentes pour prédire la puissance
features = ['Cylindrée', 'Puissance fiscale', 'Année', 'Kilométrage', 'Marque', 'Modèle']
target = 'Puissance (ch.din)'


# 4. Entraînement du modèle
X = df_train[features]
y = df_train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred=model.predict(X_val)

# Évaluation du modèle
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")

# 5. Prédiction des valeurs manquantes
X_missing = df_missing[features]
df_missing.loc[:, 'Puissance (ch.din)'] = model.predict(X_missing).astype(int)


# 6. Remettre les valeurs prédites dans le DataFrame original
df.loc[df["Puissance (ch.din)"] == 0, 'Puissance (ch.din)'] = df_missing['Puissance (ch.din)']


# 7. Décoder les colonnes catégoriques
for col in categorical_columns:
    le = label_encoders[col]
    df[col] = le.inverse_transform(df[col])

# 8. Sauvegarde des données
df.to_csv('car_data_imputed_ordered.csv', index=False)
print("Les données ont été sauvegardées dans car_data_imputed_ordered.csv avec l'ordre d'origine.")
