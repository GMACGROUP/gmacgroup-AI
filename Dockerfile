FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY gmac_group_knowledge_base.txt .
COPY static/ static/
COPY templates/ templates/

# Expose port
EXPOSE 7860

# Run with Gunicorn using dynamic PORT for Railway/Heroku/Render
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} app:app
