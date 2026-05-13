# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────
# Chargement des variables d'environnement
# ─────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────
# Initialisation Groq
# ─────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

if groq_client:
    print("✅ Client Groq initialisé avec succès.")
else:
    print("⚠️ GROQ_API_KEY non trouvée. Le bouton 'Expliquer' affichera un message d'erreur.")

# ─────────────────────────────────────────
# Prompt système par défaut
# ─────────────────────────────────────────
SYSTEM_PROMPT_DEFAULT = """Tu es un assistant médical sénégalais expert. 
Tu reçois un diagnostic et des données patient de l'application SénSanté.
Explique le résultat en français simple, mais en y intégrant des expressions en wolof 
pour être plus proche du patient (ex: "Ndank ndank", "Dina bakh", "Consultel docteur").
Sois très rassurant mais rappelle qu'il faut impérativement aller au centre de santé.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-même."""

# ─────────────────────────────────────────
# Schémas Pydantic (Rectifiés)
# ─────────────────────────────────────────
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=34.0, le=43.0) # Plage élargie
    tension_sys: int = Field(..., ge=40, le=400)    # Augmenté à 400 pour éviter l'erreur 422
    toux: bool
    fatigue: bool
    maux_tete: bool
    region: str

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str
    system_prompt: str = None

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# ─────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────
app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Fichiers statiques (Frontend)
# ─────────────────────────────────────────
# Assure-toi que le dossier 'frontend' existe à la racine du projet
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ─────────────────────────────────────────
# Chargement du modèle ML
# ─────────────────────────────────────────
try:
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    print(f"✅ Modèles chargés. Classes détectées : {list(model.classes_)}")
except Exception as e:
    print(f"❌ Erreur critique de chargement des modèles : {e}")
    model = None

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Servir l'interface utilisateur."""
    index_path = "frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Fichier frontend/index.html introuvable"}

@app.get("/health")
def health_check():
    return {"status": "ok", "ml_model_loaded": model is not None}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # Log pour débugger dans la console Hugging Face
    print(f"📥 Requête reçue - Age: {patient.age}, Tension: {patient.tension_sys}, Région: {patient.region}")

    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle d'intelligence artificielle n'est pas prêt.")

    try:
        # Encodage des données catégorielles
        sexe_enc = le_sexe.transform([patient.sexe])[0]
        region_enc = le_region.transform([patient.region])[0]
        
        # Préparation du vecteur pour le modèle
        features = np.array([[
            patient.age,
            sexe_enc,
            patient.temperature,
            patient.tension_sys,
            int(patient.toux),
            int(patient.fatigue),
            int(patient.maux_tete),
            region_enc
        ]])

        # Prédiction
        diagnostic = model.predict(features)[0]
        proba_max = float(np.max(model.predict_proba(features)))

        confiance = (
            "haute" if proba_max >= 0.7
            else "moyenne" if proba_max >= 0.4
            else "faible"
        )

        messages = {
            "palu": "Suspicion de paludisme. Consultez rapidement un médecin pour un test TDR.",
            "grippe": "Symptômes grippaux détectés. Repos et beaucoup d'eau.",
            "typh": "Suspicion de typhoïde. Une analyse de sang (Widal) est recommandée.",
            "sain": "Pas de pathologie majeure détectée d'après ces symptômes."
        }

        return DiagnosticOutput(
            diagnostic=diagnostic,
            probabilite=round(proba_max, 2),
            confiance=confiance,
            message=messages.get(diagnostic, "Veuillez consulter un professionnel de santé.")
        )
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de traitement : {str(e)}")

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(explication="Le service d'explication est temporairement indisponible (Clé API manquante sur le serveur).")

    prompt_actif = data.system_prompt if data.system_prompt else SYSTEM_PROMPT_DEFAULT

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}. "
        f"Température : {data.temperature}°C. "
        f"Résultat prédit : {data.diagnostic} ({data.probabilite:.0%}). "
        f"Explique ce résultat avec bienveillance."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt_actif},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.4
        )
        return ExplainOutput(explication=response.choices[0].message.content)
    except Exception as e:
        return ExplainOutput(explication=f"Dina bakh, mais nous avons un souci avec l'IA : {str(e)}")