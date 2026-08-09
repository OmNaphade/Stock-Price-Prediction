# python:3.10-slim always resolves to the latest 3.10.x patch (currently 3.10.16)
# DO NOT use 3.10.20 — that tag does not exist on Docker Hub
FROM python:3.10-slim

WORKDIR /app

# System deps needed by curl_cffi and matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Litestream: continuous SQLite replication to object storage, only
# activated at runtime if LITESTREAM_BUCKET_URL is set (see entrypoint.sh).
# Bump LITESTREAM_VERSION independently of the Python deps above.
# NOTE: asset naming has changed across releases — no "v" prefix before
# the version in the filename (only in the release tag/URL path), and
# the arch is "x86_64", not "amd64" (that older naming now belongs to a
# different "litestream-vfs-*" build variant). Verify against
# https://github.com/benbjohnson/litestream/releases before bumping this.
ARG LITESTREAM_VERSION=0.5.16
RUN curl -fsSL -o /tmp/litestream.tar.gz \
    "https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-${LITESTREAM_VERSION}-linux-x86_64.tar.gz" \
    && tar -C /usr/local/bin -xzf /tmp/litestream.tar.gz litestream \
    && rm /tmp/litestream.tar.gz

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .
RUN chmod +x entrypoint.sh

EXPOSE 8501

# $PORT is set by Render automatically; fall back to 8501 locally
# XSRF and CORS protection are left at Streamlit's secure defaults (enabled).
CMD ["/app/entrypoint.sh"]
