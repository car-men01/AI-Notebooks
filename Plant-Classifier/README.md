# Plant Classifier App - Complete Setup Guide

A full-stack plant classification application with AI-powered plant care recommendations using LLM agents and RAG.

## Features

- 🌿 **Plant Classification**: Upload images to identify plant species
- 🤖 **3 LLM Agent Methods**: 
  - Direct LLM generation
  - Web search enhanced
  - Combined approach
- 📚 **RAG (Retrieval Augmented Generation)**: Vector store-powered care recommendations
- 🎨 **Modern React UI**: Beautiful responsive interface

## Prerequisites

- Python 3.12+
- Node.js 16+
- OpenAI API Key
- Tavily API Key (optional, for web search)

## Project Structure

```
Plant-Classifier/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app with endpoints
│   │   ├── model.py           # Model loading
│   │   ├── predict.py         # Prediction logic
│   │   ├── schema.py          # Pydantic schemas
│   │   ├── config.py          # Configuration
│   │   ├── LLM_access/        # LLM agent implementations
│   │   └── RAG_access/        # RAG implementation
│   └── scripts/
│       └── populate_vector_store.py
├── frontend/                   # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.js
│   └── package.json
├── model/                      # Trained models (at root)
│   ├── efficientnet_b0_plants.pt
│   └── preprocessing_params.joblib
├── resources/                  # Vector store (at root)
│   └── vector_store/
├── requirements.txt
├── start_backend.ps1          # Backend startup script
└── start_frontend.ps1         # Frontend startup script
```

## Quick Start

### 1. Setup Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here
```

### 2. Run Backend

**Option A: Using startup script (recommended)**
```powershell
.\start_backend.ps1
```

**Option B: Manual start**
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start backend from backend directory
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Backend will run at http://localhost:8080

### 3. Run Frontend

**In a separate terminal:**

**Option A: Using startup script (recommended)**
```powershell
.\start_frontend.ps1
```

**Option B: Manual start**
```powershell
cd frontend
npm install
npm start
```

Frontend will open at http://localhost:3000

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /classes` - Get available plant classes
- `POST /predict` - Basic classification
- `POST /plant-care` - Classification + 3 agent care cards
- `POST /plant-care-rag` - Classification + RAG care card

## Usage

1. **Open the app** at http://localhost:3000
2. **Upload a plant image** (click or drag-and-drop)
3. **Select mode**:
   - **Basic**: Just classification
   - **Full (3 Agents)**: Classification + 3 care cards
   - **RAG**: Classification + RAG-powered care card
4. **Click "Classify Plant"**
5. **View results** with care recommendations

## Building Vector Store (Optional)

If you want to use the RAG endpoint, populate the vector store:

```powershell
cd backend
python scripts/populate_vector_store.py
```

This will create the vector store at `resources/vector_store/`.

## Docker Deployment

```powershell
# Build image
docker build -t plant-classifier .

# Run container
docker run -p 8080:8080 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e TAVILY_API_KEY=$env:TAVILY_API_KEY `
  plant-classifier
```

## Troubleshooting

### Backend fails to start - "FileNotFoundError"
- Make sure you run the backend from the `backend/` directory
- Or use the provided `start_backend.ps1` script

### Frontend can't connect to backend
- Verify backend is running at http://localhost:8080
- Check backend health: http://localhost:8080/health
- Check browser console for CORS errors

### RAG endpoint returns error
- Build the vector store first using `populate_vector_store.py`
- Check that `resources/vector_store/` exists

### API key errors
- Ensure `.env` file exists in project root
- Verify environment variables are set correctly
- Check that backend loaded the environment variables (check logs)

## Technology Stack

**Backend:**
- FastAPI 0.104.1
- PyTorch 2.2.0
- EfficientNet-B0 (pre-trained)
- LangChain 0.3.0+
- LangGraph 0.2.0+
- LanceDB 0.8.2
- OpenAI GPT-4o-mini

**Frontend:**
- React 18.2
- Axios
- CSS3

## Development

### Backend Development
```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Frontend Development
```powershell
cd frontend
npm start
```

### Adding New Plant Classes
1. Retrain the model with new plant species
2. Update `model/preprocessing_params.joblib`
3. Replace `model/efficientnet_b0_plants.pt`

### Extending RAG Knowledge Base
Edit `backend/scripts/populate_vector_store.py` and add more URLs to the `plant_urls` dictionary.

## License

MIT

## Support

For issues or questions, please check the troubleshooting section above.
