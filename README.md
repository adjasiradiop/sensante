---
title: Sensante
emoji: 🏥
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# SenSante
Assistant de pre-diagnostic medical pour le Senegal.

## Description
SenSante utilise le Machine Learning pour aider au pre-diagnostic des maladies courantes (paludisme, grippe, typhoïde) à partir des symptômes du patient.

## Structure du projet
- `data/` : Données patients (CSV)
- `models/` : Modèle ML sérialisé
- `api/` : API FastAPI
- `frontend/` : Interface web
- `notebooks/` : Scripts d'exploration