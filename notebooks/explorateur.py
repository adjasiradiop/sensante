""" 
SenSante - Exploration du dataset patients_dakar
Lab 1 : Git, Python et Structure Projet 
""" 
import pandas as pd 

# ===== CHARGER LES DONNEES ===== 
# Note : J'ai mis .xlsx car c'est ce qu'on voyait sur ta capture d'écran
# Si tu as un message d'erreur, installe l'outil avec : pip install openpyxl
df = pd.read_excel("data/patients_dakar.xlsx") 

# ===== PREMIERS APERÇUS ===== 
print("=" * 50) 
print(" SENSANTE - Exploration du dataset ") 
print("=" * 50) 

# Dimensions du dataset 
print(f"\nNombre de patients : {len(df)}") 
print(f"Nombre de colonnes : {df.shape[1]}") 
print(f"Colonnes : {list(df.columns)}") 

# Aperçu des 5 premiers patients
print(f"\n--- 5 premiers patients ---") 
print(df.head()) 

# ===== STATISTIQUES DE BASE ===== 
print(f"\n--- Statistiques descriptives ---") 
print(df.describe().round(2)) 

# ===== REPARTITION DES DIAGNOSTICS ===== 
print(f"\n--- Repartition des diagnostics ---") 
# On enlève les espaces dans "diagnostic"
diag_counts = df["diagnostic"].value_counts() 
for diag, count in diag_counts.items(): 
    pct = (count / len(df)) * 100 
    print(f"{str(diag):12s} : {count:3d} patients ({pct:.1f}%)") 

# ===== REPARTITION PAR REGION ===== 
print(f"\n--- Repartition par region (top 5) ---") 
region_counts = df["region"].value_counts().head(5) 
for region, count in region_counts.items(): 
    print(f"{str(region):15s} : {count:3d} patients") 

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC ===== 
print(f"\n--- Temperature moyenne par diagnostic ---") 
temp_by_diag = df.groupby("diagnostic")["temperature"].mean() 
for diag, temp in temp_by_diag.items(): 
    print(f"{str(diag):12s} : {temp:.1f} °C") 

print(f"\n{'=' * 50}") 
print(" Exploration terminee !") 
print(" Prochain lab : entrainer un modele ML") 
print(f"{'=' * 50}")