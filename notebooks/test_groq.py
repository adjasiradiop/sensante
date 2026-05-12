# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────
# 1. Chargement de la clé API
# ─────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvée dans .env")
    exit()

# ─────────────────────────────────────────
# 2. Initialisation du client Groq
# ─────────────────────────────────────────
client = Groq(api_key=api_key)

# ─────────────────────────────────────────
# 3. Appel à l'API — question médicale
# ─────────────────────────────────────────
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant médical sénégalais. "
                "Réponds en français simple. "
                "Maximum 3 phrases."
            ),
        },
        {
            "role": "user",
            "content": "Quels sont les symptômes du paludisme ?",
        },
    ],
    max_tokens=200,
    temperature=0.3,
)

# ─────────────────────────────────────────
# 4. Affichage des résultats
# ─────────────────────────────────────────
print("=" * 40)
print("   Réponse de Llama 3")
print("=" * 40)
print(response.choices[0].message.content)
print("-" * 40)
print(f"Tokens utilisés : {response.usage.total_tokens}")

# ─────────────────────────────────────────
# 5. Test format SénSanté — Étape 3.3
# ─────────────────────────────────────────
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant médical sénégalais.\n"
                "Tu reçois un diagnostic et des données patient.\n"
                "Explique le résultat en français simple,\n"
                "comme un médecin parlerait à son patient.\n"
                "Sois rassurant mais recommande une consultation.\n"
                "Maximum 3 phrases.\n"
                "Ne fais JAMAIS de diagnostic toi-même."
            ),
        },
        {
            "role": "user",
            "content": (
                "Patient      : Femme, 28 ans, région Dakar\n"
                "Symptômes    : température 39.5, toux, fatigue, maux de tête\n"
                "Diagnostic   : paludisme (probabilité 72%)\n"
                "Explique ce résultat au patient."
            ),
        },
    ],
    max_tokens=200,
    temperature=0.3,
)

# ─────────────────────────────────────────
# 6. Affichage — Explication SénSanté
# ─────────────────────────────────────────
print("=" * 40)
print("   Explication SénSanté")
print("=" * 40)
print(response2.choices[0].message.content)
print("-" * 40)