FROM python:3.11.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    curl \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download and install Qdrant v1.16.0
RUN wget https://github.com/qdrant/qdrant/releases/download/v1.16.0/qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz \
    && mv qdrant /usr/local/bin/qdrant \
    && rm -rf qdrant-x86_64-unknown-linux-gnu.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 7860
EXPOSE 8000
EXPOSE 6333

# Run Qdrant + FastAPI + Streamlit
CMD qdrant --storage-dir qdrant_data --uri http://0.0.0.0:6333 & \
    sleep 2 && \
    uvicorn main:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 7860