# Full-Stack Docker Deployment Guide

## Overview

This deploys both frontend (React) and backend (FastAPI) in a single Docker container using:
- **nginx** - Serves the React frontend on port 80
- **FastAPI** - Backend API on internal port 8080
- **nginx reverse proxy** - Routes `/api/*` requests to the backend

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```powershell
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access the app at: **http://localhost**

### Option 2: Using Docker directly

```powershell
# Build
docker build -f Dockerfile.fullstack -t plant-classifier-fullstack .

# Run
docker run -d -p 80:80 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e TAVILY_API_KEY=$env:TAVILY_API_KEY `
  --name plant-app `
  plant-classifier-fullstack

# View logs
docker logs -f plant-app
```

Access the app at: **http://localhost**

## How It Works

### Architecture

```
[Browser] → [nginx:80]
                ├─→ / (frontend static files)
                └─→ /api/* → [FastAPI:8080]
```

1. **Frontend**: React app is built and served by nginx
2. **Backend**: FastAPI runs on port 8080 (internal)
3. **Routing**: nginx proxies `/api/*` requests to the backend

### File Structure in Container

```
/app/
├── frontend/build/        # Built React app
├── backend/app/           # FastAPI application
├── model/                 # ML models
└── resources/             # Vector store
```

## Prerequisites

Before building, make sure:

1. **Vector store is populated**:
   ```powershell
   cd backend
   python scripts/populate_vector_store.py
   cd ..
   ```

2. **Environment variables are set**:
   ```powershell
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   echo "TAVILY_API_KEY=your-key-here" >> .env
   ```

3. **Frontend builds successfully**:
   ```powershell
   cd frontend
   npm install
   npm run build
   cd ..
   ```

## Common Commands

### Docker Compose

```powershell
# Start
docker-compose up -d

# Rebuild and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Remove volumes
docker-compose down -v
```

### Docker

```powershell
# Build
docker build -f Dockerfile.fullstack -t plant-classifier-fullstack .

# Run
docker run -d -p 80:80 --env-file .env --name plant-app plant-classifier-fullstack

# Stop
docker stop plant-app

# Remove
docker rm plant-app

# View logs
docker logs -f plant-app

# Execute commands inside container
docker exec -it plant-app bash
```

## Customization

### Change Port

To run on a different port (e.g., 8000):

```powershell
# Docker Compose - edit docker-compose.yml
ports:
  - "8000:80"

# Docker run
docker run -d -p 8000:80 --env-file .env plant-classifier-fullstack
```

Access at: http://localhost:8000

### Environment Variables

Add more environment variables:

```yaml
# docker-compose.yml
environment:
  - OPENAI_API_KEY=${OPENAI_API_KEY}
  - TAVILY_API_KEY=${TAVILY_API_KEY}
  - CUSTOM_VAR=${CUSTOM_VAR}
```

### Custom nginx Configuration

Edit the nginx config in `Dockerfile.fullstack`:

```dockerfile
RUN echo 'server { \n\
    listen 80; \n\
    # Your custom configuration \n\
} \n\
' > /etc/nginx/sites-available/default
```

## Troubleshooting

### Port 80 already in use

Use a different port:
```powershell
docker run -d -p 8080:80 --env-file .env plant-classifier-fullstack
```

### Frontend shows blank page

Check nginx logs:
```powershell
docker exec plant-app cat /var/log/nginx/error.log
```

Verify build folder exists:
```powershell
docker exec plant-app ls -la /app/frontend/build
```

### API requests fail

Check backend logs:
```powershell
docker logs plant-app | grep "INFO:"
```

Test backend directly:
```powershell
docker exec plant-app curl http://localhost:8080/health
```

### nginx not starting

Check nginx configuration:
```powershell
docker exec plant-app nginx -t
```

### Frontend can't reach backend

Verify nginx proxy configuration:
```powershell
docker exec plant-app cat /etc/nginx/sites-available/default
```

Test API endpoint:
```powershell
curl http://localhost/api/health
```

## Production Recommendations

### 1. Use Multi-Stage Build (Already Implemented)

The Dockerfile uses multi-stage builds to keep the image small:
- Stage 1: Build frontend (Node.js)
- Stage 2: Run application (Python + nginx)

### 2. Health Checks

Add to docker-compose.yml:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 3. Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 4. Persistent Logs

Mount log volumes:
```yaml
volumes:
  - ./logs:/var/log/nginx
```

### 5. SSL/HTTPS

For production, use a reverse proxy like Caddy or Traefik with automatic SSL.

### 6. Environment-Specific Builds

Create separate Dockerfiles for dev/prod:
- `Dockerfile.dev` - Development with hot reload
- `Dockerfile.fullstack` - Production build

## Debugging

### Access Container Shell

```powershell
docker exec -it plant-app bash
```

### Check Running Processes

```powershell
docker exec plant-app ps aux
```

### Test Backend Inside Container

```powershell
docker exec plant-app curl http://localhost:8080/health
```

### Test Frontend Inside Container

```powershell
docker exec plant-app curl http://localhost
```

### View All Logs

```powershell
# Application logs
docker logs -f plant-app

# nginx access logs
docker exec plant-app tail -f /var/log/nginx/access.log

# nginx error logs
docker exec plant-app tail -f /var/log/nginx/error.log
```

## Updates and Maintenance

### Update Application

```powershell
# Pull latest code
git pull

# Rebuild
docker-compose down
docker-compose up -d --build
```

### Update Dependencies

```powershell
# Update Python packages
pip freeze > requirements.txt

# Update Node packages
cd frontend
npm update
cd ..

# Rebuild
docker-compose up -d --build
```

## Performance Optimization

1. **nginx caching** - Add caching headers for static files
2. **gzip compression** - Enable in nginx config
3. **CDN** - Serve static assets from CDN
4. **Image optimization** - Compress images before upload
5. **Backend caching** - Add Redis for API response caching

## Security

1. **Don't expose port 8080** - Only backend port 80 should be accessible
2. **Use secrets management** - Don't commit .env files
3. **Regular updates** - Keep base images updated
4. **Scan images** - Use Docker Scout or Trivy
5. **Limit resources** - Set memory/CPU limits

## Complete Example

```powershell
# 1. Populate vector store
cd backend
python scripts/populate_vector_store.py
cd ..

# 2. Set environment variables
echo "OPENAI_API_KEY=sk-..." > .env
echo "TAVILY_API_KEY=tvly-..." >> .env

# 3. Build and run
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Access app
# Open browser: http://localhost

# 6. Stop when done
docker-compose down
```

That's it! Your full-stack plant classifier is now running in Docker! 🐳🌿
