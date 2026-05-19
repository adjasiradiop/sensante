---
title: Sensante
emoji: 🏥
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# SenSante 🏥
Assistant de pré-diagnostic médical pour le Sénégal.

## 🌐 Démo en ligne
👉 [https://adjaaaa-sensante.hf.space](https://adjaaaa-sensante.hf.space)

## 📋 Description
SenSante utilise le Machine Learning pour aider au pré-diagnostic des maladies courantes (paludisme, grippe, typhoïde) à partir des symptômes du patient. L'application intègre également une explication en français et en wolof grâce à Llama 3.

## 🛠️ Stack technique
- **scikit-learn** — Modèle ML (Random Forest)
- **FastAPI** — API REST
- **Tailwind CSS** — Frontend responsive
- **Groq / Llama 3** — Explication LLM en français/wolof
- **Docker** — Conteneurisation

## 📁 Structure du projet
```
sensante/
├── data/          → Données patients (CSV)
├── models/        → Modèle ML sérialisé (.pkl)
├── api/           → API FastAPI
├── frontend/      → Interface web
└── notebooks/     → Scripts d'exploration
```

## 👤 Auteur
**Adja Sira Diop** — L2 GLSI — ESP/UCAD — 2026