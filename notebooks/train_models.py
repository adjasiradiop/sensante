import pandas as pd
import numpy as np
# Charger le dataset
df = pd . read_csv (" data / patients_dakar . csv")
# Verifier les dimensions
print ( f"Dataset : {df. shape [0]} patients , {df. shape [1]} colonnes ")
print ( f"\nColonnes : { list (df. columns )}")
print ( f"\nDiagnostics :\n{df[ ' diagnostic ']. value_counts ()}")
from sklearn . preprocessing import LabelEncoder
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder ()
le_region = LabelEncoder ()
df [' sexe_encoded '] = le_sexe . fit_transform ( df ['sexe '])
df [' region_encoded '] = le_region . fit_transform ( df ['region '])
# Definir les features (X) et la cible (y)
feature_cols = ['age ', ' sexe_encoded ', 'temperature ', 'tension_sys ',
'toux ', 'fatigue ', 'maux_tete ', ' region_encoded ']

X = df [ feature_cols ]
y = df ['diagnostic ']
print ( f" Features : {X. shape }") # (500 , 8)
print ( f" Cible : {y. shape }") # (500 ,)
from sklearn . model_selection import train_test_split
# 80% pour l'entrainement , 20% pour le test
X_train , X_test , y_train , y_test = train_test_split (
X , y ,
test_size =0.2 , # 20% pour le test
random_state =42 , # Pour avoir les memes resultats a chaque fois
stratify = y # Garder les memes proportions de diagnostics
)
print ( f" Entrainement : { X_train . shape [0]} patients ")
print ( f" Test : { X_test . shape [0]} patients ")