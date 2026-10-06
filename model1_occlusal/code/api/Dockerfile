# CPU inference image for clinic PCs / servers.  Build from repo root:
#   docker build -f api/Dockerfile -t dentassist-opg .
#   docker run -p 8000:8000 -v $PWD/weights:/app/weights dentassist-opg
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dentassist ./dentassist
COPY api ./api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
