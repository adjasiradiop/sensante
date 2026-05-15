"""
SenSante - Exploration du dataset patients_dakar_500.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd
import os

# ==============================
# 1. GESTION DU CHEMIN (Plus robuste)
# ==============================
# Utiliser os.path permet d'éviter les erreurs selon si tu lances le script 
# depuis la racine ou depuis le dossier 'scripts'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On pointe vers le nouveau fichier de 500 patients
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "patients_dakar_500.csv")

# ==============================
# 2. CHARGER LES DONNEES
# ==============================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Erreur : Le fichier est introuvable à l'adresse : {DATA_PATH}")
    print("Vérifiez que le fichier 'patients_dakar_500.csv' est bien dans le dossier 'data'.")
    exit()

# ==============================
# 3. PREMIERS APERÇUS
# ==============================
print("=" * 50)
print(" SENSANTE - Exploration du dataset (v500)")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {', '.join(df.columns)}")

# Aperçu des 5 premières lignes
print(f"\n--- 5 premiers patients ---")
print(df.head())

# ==============================
# 4. STATISTIQUES DE BASE
# ==============================
print(f"\n--- Statistiques descriptives ---")
# On arrondit pour la lisibilité
print(df.describe().round(2))

# ==============================
# 5. ANALYSE DES SYMPTÔMES CLÉS
# ==============================
# Puisque c'est pour SenSante, analysons les nouveaux symptômes
print(f"\n--- Analyse des nouveaux symptômes (%) ---")
for col in ['toux', 'fatigue', 'maux_tete', 'frissons', 'nausee']:
    presence = df[col].mean() * 100
    print(f" {col:10s} : {presence:>5.1f}% des patients")

# ==============================
# 6. REPARTITION DES DIAGNOSTICS
# ==============================
print(f"\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f" {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ==============================
# 7. TEMPERATURE MOYENNE PAR DIAGNOSTIC
# ==============================
print(f"\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f" {diag:12s} : {temp:.1f} °C")

print(f"\n{'=' * 50}")
print("Exploration terminée ! Prêt pour le Machine Learning.")
print(f"{'=' * 50}")