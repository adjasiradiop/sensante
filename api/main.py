# api/main.py

from fastapi import FastAPI
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

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print(" Client Groq initialisé.")
else:
    print(" GROQ_API_KEY non trouvée. /explain désactivé.")

# ─────────────────────────────────────────
# Prompt système
# ─────────────────────────────────────────

SYSTEM_PROMPT = """
Tu es un assistant medical senegalais.
Tu expliques un diagnostic fourni.
Tu utilises un francais simple.
Tu es rassurant.
Tu recommandes toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
"""

# ─────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
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


class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"


# ─────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────

app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.3.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Chargement du modèle ML
# ─────────────────────────────────────────

print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")

print(f"Modele charge : {list(model.classes_)}")

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}


@app.get("/model-info")
def model_info():
    return {
        "type": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encodage
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
        region_enc = le_region.transform([patient.region])[0]
    except ValueError as e:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=str(e)
        )

    # Features
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
    proba_max = float(model.predict_proba(features)[0].max())

    confiance = (
        "haute" if proba_max >= 0.7
        else "moyenne" if proba_max >= 0.4
        else "faible"
    )

    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoide. Consultation necessaire.",
        "sain": "Pas de pathologie detectee."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )


@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):

    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree."
        )

    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic : {data.diagnostic} "
        f"(probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat simplement."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(
        explication=explication
    )