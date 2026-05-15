# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────
# 1. CONFIGURATION & ENVIRONNEMENT
# ─────────────────────────────────────────
load_dotenv()

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.4.7"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# 2. INITIALISATION IA (GROQ)
# ─────────────────────────────────────────
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        print(f"GROQ INITIALISÉ : Clé détectée (début : {groq_api_key[:6]}...)")
    except Exception as e:
        print(f"ERREUR INITIALISATION GROQ : {e}")
        groq_client = None
else:
    groq_client = None
    print("⚠️ GROQ NON INITIALISÉ : Variable GROQ_API_KEY introuvable.")

SYSTEM_PROMPT_DEFAULT = """Tu es un assistant médical sénégalais expert. 
Tu reçois un diagnostic et des données patient de l'application SénSanté.
Explique le résultat en français simple, mais en y intégrant des expressions en wolof 
pour être plus proche du patient (ex: "Ndank ndank", "Dina bakh", "Consultel docteur").
Sois très rassurant mais rappelle qu'il faut impérativement aller au centre de santé.
Maximum 3 phrases. Ne fais JAMAIS de diagnostic toi-même."""

# ─────────────────────────────────────────
# 3. CHARGEMENT DES MODÈLES (CHEMINS ROBUSTES)
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

# Débogage : Liste les fichiers pour vérifier la casse (minuscules/majuscules)
if MODELS_DIR.exists():
    print(f"📂 Contenu du dossier models : {os.listdir(MODELS_DIR)}")
else:
    print("❌ DOSSIER MODELS INTROUVABLE !")

try:
    # On charge les 3 fichiers indispensables
    model = joblib.load(MODELS_DIR / "model.pkl")
    le_sexe = joblib.load(MODELS_DIR / "encoder_sexe.pkl")
    le_region = joblib.load(MODELS_DIR / "encoder_region.pkl")
    print(f"✅ MODÈLES ML : Chargés avec succès. Classes : {list(model.classes_)}")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE CHARGEMENT MODÈLES : {e}")
    model = None

# ─────────────────────────────────────────
# 4. SCHÉMAS DE DONNÉES (PYDANTIC)
# ─────────────────────────────────────────
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str
    temperature: float = Field(..., ge=34.0, le=43.0)
    tension_sys: int = Field(..., ge=40, le=400)
    toux: bool
    fatigue: bool
    maux_tete: bool
    frissons: bool
    nausee: bool
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

# ─────────────────────────────────────────
# 5. ROUTES API
# ─────────────────────────────────────────

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Le modèle d'intelligence artificielle n'est pas prêt.")

    try:
        # Encodage (Attention : doit correspondre aux valeurs d'entraînement)
        sexe_enc = le_sexe.transform([patient.sexe])[0]
        region_enc = le_region.transform([patient.region])[0]
        
        features = np.array([[
            patient.age, sexe_enc, patient.temperature, patient.tension_sys,
            int(patient.toux), int(patient.fatigue), int(patient.maux_tete),
            int(patient.frissons), int(patient.nausee), region_enc
        ]])

        diag = model.predict(features)[0]
        proba = float(np.max(model.predict_proba(features)))
        confiance = "haute" if proba >= 0.7 else "moyenne" if proba >= 0.4 else "faible"

        messages = {
            "paludisme": "Suspicion de paludisme. Consultez rapidement pour un test TDR.",
            "grippe": "Symptômes grippaux. Repos et hydratation conseillés.",
            "typhoide": "Suspicion de typhoïde. Une analyse de sang est recommandée.",
            "sain": "Pas de pathologie majeure détectée. Restez vigilant."
        }

        return DiagnosticOutput(
            diagnostic=diag,
            probabilite=round(proba, 2),
            confiance=confiance,
            message=messages.get(diag, "Consultez un professionnel de santé.")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain(data: ExplainInput):
    if not groq_client:
        return {"explication": "Service d'IA indisponible (Clé manquante)."}

    prompt_actif = data.system_prompt or SYSTEM_PROMPT_DEFAULT
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, {data.region}. "
        f"Température : {data.temperature}°C. Diagnostic : {data.diagnostic}."
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": prompt_actif},
                {"role": "user", "content": user_prompt}
            ]
        )
        return {"explication": completion.choices[0].message.content}
    except Exception as e:
        return {"explication": f"Dina bakh, souci technique : {str(e)}"}

# ─────────────────────────────────────────
# 6. GESTION DU FRONTEND & SANTÉ
# ─────────────────────────────────────────
FRONTEND_DIR = ROOT_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Frontend non trouvé."}

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "model_ready": model is not None,
        "groq_ready": groq_client is not None
    }