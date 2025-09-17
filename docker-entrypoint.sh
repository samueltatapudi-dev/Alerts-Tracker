#!/bin/sh
set -e

: "${PORT:=5000}"

python - <<'PY'
import time
from app import app, db, ensure_email_settings, apply_email_settings

max_attempts = 5
sleep_seconds = 1.0
for attempt in range(1, max_attempts + 1):
    try:
        with app.app_context():
            db.create_all()
            settings = ensure_email_settings()
            apply_email_settings(settings)
    except Exception as exc:
        if attempt == max_attempts:
            raise
        time.sleep(sleep_seconds)
        sleep_seconds *= 2
    else:
        break
PY

exec gunicorn --bind "0.0.0.0:${PORT}" app:app
