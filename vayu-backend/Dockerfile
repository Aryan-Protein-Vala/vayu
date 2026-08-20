# VAYU backend — deployable container (HF Spaces / Render / Railway)
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only (lightweight, fast cold start)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + prebuilt data (index.faiss, chunks.json, parents.json, vectorizer.pkl)
COPY backend/ backend/
COPY data/ data/
COPY benchmark_results.json .

ENV VAYU_ENV=production
EXPOSE 8000

CMD ["python", "-m", "backend.main"]
