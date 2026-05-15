import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# 1. CHEMIN AUTOMATIQUE DU PROJET
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Mise à jour avec le nom du nouveau fichier CSV
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "patients_dakar_500.csv")

# ==============================
# 2. CHARGEMENT DU DATASET
# ==============================

try:
    df = pd.read_csv(DATA_PATH)
    print(f"Fichier {os.path.basename(DATA_PATH)} chargé avec succès !")
except Exception as e:
    print(f"Erreur lors de la lecture du fichier : {e}")
    print(f"Chemin utilisé : {DATA_PATH}")
    exit()

# ==============================
# 3. PRÉTRAITEMENT
# ==============================

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# Ajout des nouvelles colonnes présentes dans le fichier de 500 patients
feature_cols = [
    'age', 'sexe_encoded', 'temperature',
    'tension_sys', 'toux', 'fatigue',
    'maux_tete', 'frissons', 'nausee', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

# ==============================
# 4. TRAIN / TEST
# ==============================

# Le paramètre stratify=y est très important ici pour garder l'équilibre des maladies
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. ENTRAÎNEMENT
# ==============================

# On augmente un peu la profondeur pour mieux apprendre sur 500 lignes
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Modèle entraîné sur {len(df)} patients !")
print(f"Précision : {accuracy_score(y_test, model.predict(X_test)):.2%}")

# ==============================
# 6. SAUVEGARDE
# ==============================

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(le_sexe, os.path.join(MODELS_DIR, "encoder_sexe.pkl"))
joblib.dump(le_region, os.path.join(MODELS_DIR, "encoder_region.pkl"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))

print("Tous les fichiers modèles (v500) ont été sauvegardés correctement.")