FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 5000
CMD ["sh", "-c", "python -c \"from pathlib import Path; Path('instance').mkdir(exist_ok=True); from app import app, db; ctx = app.app_context(); ctx.push(); db.create_all()\" && exec gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]
