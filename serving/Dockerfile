FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY pyproject.toml .
COPY src/ src/
COPY serving/app.py serving/app.py
COPY feature_store/ feature_store/
# models/artifacts is gitignored (binary model files) -- mounted as a volume at runtime, not
# baked into the image, so a retrained model doesn't require a rebuild.

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
