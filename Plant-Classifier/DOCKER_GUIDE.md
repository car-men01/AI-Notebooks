# Docker Deployment Guide

## Prerequisites

1. Docker Desktop installed and running
2. Vector store populated (run `backend/scripts/populate_vector_store.py` first)
3. API keys ready (OPENAI_API_KEY, TAVILY_API_KEY)

## Build the Docker Image

From the project root directory:

```powershell
docker build -t plant-classifier .
```

This will:
- Use Python 3.12-slim base image
- Install all dependencies from requirements.txt
- Copy model files, resources (vector store), and backend code
- Set up the application to run from the backend directory

## Run the Container

### Basic Run (with environment variables)

```powershell
docker run -p 8080:8080 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e TAVILY_API_KEY=$env:TAVILY_API_KEY `
  plant-classifier
```

### Run with .env file

```powershell
docker run -p 8080:8080 --env-file .env plant-classifier
```

### Run in detached mode (background)

```powershell
docker run -d -p 8080:8080 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e TAVILY_API_KEY=$env:TAVILY_API_KEY `
  --name plant-classifier-app `
  plant-classifier
```

## Verify the Container

Check if it's running:
```powershell
docker ps
```

View logs:
```powershell
docker logs plant-classifier-app
```

Follow logs in real-time:
```powershell
docker logs -f plant-classifier-app
```

Test the API:
```powershell
curl http://localhost:8080/health
```

## Stop and Remove Container

Stop:
```powershell
docker stop plant-classifier-app
```

Remove:
```powershell
docker rm plant-classifier-app
```

## Important Notes

### 1. Vector Store Must Be Pre-built

Before building the Docker image, make sure you have populated the vector store:

```powershell
cd backend
python scripts/populate_vector_store.py
```

This creates the `resources/vector_store/` directory that gets copied into the Docker image.

### 2. Environment Variables

The container needs these environment variables:
- `OPENAI_API_KEY` - Required for LLM and RAG functionality
- `TAVILY_API_KEY` - Optional, only needed for web search agent

### 3. Port Mapping

The container exposes port 8080. You can map it to a different host port:

```powershell
# Map to port 9000 on host
docker run -p 9000:8080 -e OPENAI_API_KEY=$env:OPENAI_API_KEY plant-classifier
```

### 4. Model Files

Make sure these files exist before building:
- `model/efficientnet_b0_plants.pt`
- `model/preprocessing_params.joblib`

## Docker Compose (Optional)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    restart: unless-stopped
```

Run with:
```powershell
docker-compose up -d
```

## Troubleshooting

### Container exits immediately

Check logs:
```powershell
docker logs plant-classifier-app
```

Common issues:
- Missing API keys
- Model files not found
- Vector store not initialized

### Cannot connect to container

Verify port mapping:
```powershell
docker port plant-classifier-app
```

Check if port 8080 is already in use:
```powershell
netstat -ano | findstr :8080
```

### API keys not working

Verify they're set in the container:
```powershell
docker exec plant-classifier-app env | Select-String "API_KEY"
```

### Vector store errors

Make sure you built the vector store before creating the Docker image. If not, rebuild:
```powershell
# Build vector store
cd backend
python scripts/populate_vector_store.py
cd ..

# Rebuild Docker image
docker build -t plant-classifier .
```

## Production Deployment

For production, consider:

1. **Use a .dockerignore file** to exclude unnecessary files
2. **Multi-stage builds** to reduce image size
3. **Health checks** in Dockerfile
4. **Volume mounts** for logs
5. **Reverse proxy** (nginx) for SSL/HTTPS
6. **Environment-specific configs**

Example with health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

## Frontend Deployment

The Docker setup only includes the backend. For the frontend:

### Option 1: Serve frontend separately
```powershell
cd frontend
npm run build
# Serve the build/ folder with nginx or a static host
```

### Option 2: Add frontend to Docker
Create a multi-stage Dockerfile that builds both frontend and backend.

## Quick Reference

```powershell
# Build
docker build -t plant-classifier .

# Run
docker run -d -p 8080:8080 --name plant-app --env-file .env plant-classifier

# Logs
docker logs -f plant-app

# Stop
docker stop plant-app

# Remove
docker rm plant-app

# Restart
docker restart plant-app
```
