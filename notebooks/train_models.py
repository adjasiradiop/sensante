import pandas as pd
import numpy as np
# Charger le dataset
df = pd . read_csv (" data / patients_dakar . csv")
# Verifier les dimensions
print ( f" Dataset : {df. shape [0]} patients , {df. shape [1]} colonnes ")
print ( f"\ nColonnes : { list (df. columns )}")
print ( f"\ nDiagnostics :\n{df[ ' diagnostic ']. value_counts ()}")