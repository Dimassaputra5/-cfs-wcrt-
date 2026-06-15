FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/cfs_wcrt/ui/web.py", "--server.address=0.0.0.0", "--server.port=8501"]
