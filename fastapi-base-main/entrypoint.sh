#!/bin/sh
set -e

# Dependencies are installed during the image build.
if [ "$DEBUG" = "true" ]; then
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi

exec gunicorn app.main:app --workers 17 --worker-class uvicorn.workers.UvicornWorker --threads 8 --timeout 120 --keep-alive 5 --bind 0.0.0.0:8000
