# 1. Image de base : Python 3.11 léger
FROM python:3.11-slim

# 2. Variables d'environnement pour Python
# Empêche la création de fichiers .pyc et force l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Dossier de travail dans le conteneur
WORKDIR /app

# 4. Installation des dépendances système (si nécessaire)
# Utile pour certains paquets Python qui compilent du C (ex: psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copier et installer les dépendances
# Optimisation du cache Docker : cette couche n'est recréée que si requirements.txt change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le code du projet
COPY . .

# 7. Déclarer le port (informatif)
EXPOSE 8000

# 8. Commande de démarrage
# Utilisation de la forme "exec" (recommandée)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]