FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# ── Deployment tooling (Phase 7) ──────────────────────────────────────────────
# Install the Docker CLI (client only — it talks to the HOST daemon via the
# socket mounted in docker-compose, so no daemon runs inside this container)
# and the AWS CLI v2. Baking these into the image is more reliable than
# installing them at container startup: deterministic, no startup network
# dependency, and they're guaranteed present before Phase 7 runs.
RUN apt-get update && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# AWS CLI v2 (architecture-aware: works on amd64 and arm64)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then AWS_ARCH=aarch64; else AWS_ARCH=x86_64; fi && \
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_ARCH}.zip" -o /tmp/awscliv2.zip && \
    apt-get update && apt-get install -y unzip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    apt-get purge -y unzip && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-bake the embedding model into the image so the container never downloads
# it at runtime. This caches ~90MB of weights into the HF cache layer, so the
# first request after deploy is fast (no cold download). Keep the model name in
# sync with core/embeddings.py (EMBEDDING_MODEL).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]