FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (at root level)
COPY model/ ./model/

# Copy resources (vector store - at root level)
COPY resources/ ./resources/

# Copy backend application code
COPY backend/app/ ./backend/app/

# Copy scripts if needed
COPY backend/scripts/ ./backend/scripts/

# Set working directory to backend for running uvicorn
WORKDIR /app/backend

# Expose port
EXPOSE 8080

# Run the application from backend directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]