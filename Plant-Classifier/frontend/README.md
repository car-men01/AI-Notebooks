# Plant Care App - Frontend Setup

## Prerequisites
- Node.js (v16 or higher)
- npm (comes with Node.js)
- Backend running at http://localhost:8080

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the App

### Development Mode
Start the development server with hot-reload:
```bash
npm start
```
The app will open at http://localhost:3000

### Production Build
Build the app for production:
```bash
npm run build
```
This creates an optimized build in the `build/` folder.

## Backend Requirements

Make sure the FastAPI backend is running:

```bash
# Navigate to project root
cd ..

# Run with Docker (recommended)
docker build -t plant-classifier .
docker run -p 8080:8080 -e OPENAI_API_KEY=$env:OPENAI_API_KEY -e TAVILY_API_KEY=$env:TAVILY_API_KEY plant-classifier

# OR run locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Features

### 1. Basic Mode
- Upload plant image
- Get classification with confidence score
- View top 5 predictions

### 2. Full Mode (3 Agents)
- All basic mode features
- **Direct Agent**: LLM-generated plant care card
- **Web Search Agent**: Care card enhanced with web search data
- **Combined Agent**: Best of both worlds with web research

### 3. RAG Mode
- All basic mode features
- **RAG Agent**: Care card powered by vector store retrieval
- Uses pre-built plant care database

## Usage

1. **Upload Image**: Click or drag-and-drop a plant image
2. **Select Mode**: Choose Basic, Full (3 Agents), or RAG
3. **Classify**: Click "Classify Plant" button
4. **View Results**: 
   - Classification results with confidence
   - Plant care card(s) depending on mode
   - In Full mode, switch between agent tabs

## Troubleshooting

### Backend Connection Error
- Verify backend is running at http://localhost:8080
- Check backend health: http://localhost:8080/api/health

### CORS Issues
- Backend is configured with CORS middleware
- Frontend proxy is set to http://localhost:8080

### Image Upload Issues
- Supported formats: PNG, JPG, JPEG
- Max file size: 10MB (adjust in backend if needed)

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── ImageUpload.js
│   │   ├── ImageUpload.css
│   │   ├── Results.js
│   │   └── Results.css
│   ├── services/
│   │   └── api.js
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
└── package.json
```

## API Endpoints Used

- `GET /api/classes` - Get available plant classes
- `POST /api/predict` - Classify plant image
- `POST /api/plant-care` - Get care cards from 3 agents
- `POST /api/plant-care-rag` - Get care card from RAG
- `GET /api/health` - Health check

## Development Notes

- React 18.2 with functional components and hooks
- Axios for API calls
- CSS modules for styling
- Responsive design for mobile/tablet/desktop
