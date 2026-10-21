# Hugging Face Space (Docker SDK). Serves the FastAPI app on port 7860.
# Spaces run the container as UID 1000, so everything lives under that user's home.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown -R user:user /home/user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf
WORKDIR /home/user/app

COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

# Optional fallback model (fastText lid.176, ~900 KB compressed, 176 languages)
RUN curl -fsSL -o lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz || true

COPY --chown=user app/ ./app/

# MODEL_ID is set as a Space variable (e.g. yourname/polyglot-120)
ENV CONF_THRESHOLD=0.70 FASTTEXT_PATH=/home/user/app/lid.176.ftz
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
