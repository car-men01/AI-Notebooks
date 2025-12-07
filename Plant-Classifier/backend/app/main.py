# FastAPI app

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model import load_model
from .predict import predict_image
from .schema import PredictionResponse
from .LLM_access import (
    generate_plant_care_card_direct,
    generate_plant_care_card_web,
    generate_plant_care_card_combined,
    PlantCareCard
)
from .RAG_access import generate_plant_care_card_rag, VectorStoreManager, set_vector_store_manager
import torch
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

app = FastAPI(title="Plant Classifier API")

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
params = None
vector_store_manager = None


@app.on_event("startup")
async def startup_event():
    global model, params, vector_store_manager
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get paths relative to project root (parent of backend/)
    import pathlib
    backend_dir = pathlib.Path(__file__).parent.parent
    project_root = backend_dir.parent
    model_dir = project_root / "model"
    
    model, params = load_model(
        str(model_dir / "efficientnet_b0_plants.pt"),
        str(model_dir / "preprocessing_params.joblib"),
        device
    )
    
    # Initialize vector store for RAG
    try:
        # Use project root for resources path
        resources_dir = project_root / "resources"
        vector_store_manager = VectorStoreManager(str(resources_dir))
        vector_store_manager.initialize()
        set_vector_store_manager(vector_store_manager)  # Set global reference
        print("Vector store initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize vector store: {e}")
        print("RAG endpoint will not be available")


@app.get("/")
async def root():
    return {"message": "Plant Classifier API is running"}


@app.get("/classes")
async def get_classes():
    """Get list of all plant classes the model can recognize"""
    if params is None:
        return {"error": "Model not loaded"}

    return {
        "total_classes": len(params['classes']),
        "classes": sorted(params['classes'])
    }


@app.post("/predict")
async def classify_plant(file: UploadFile = File(...)):
    """Upload an image to get plant classification"""
    if model is None or params is None:
        return {"error": "Model not loaded"}

    # Read file contents
    contents = await file.read()
    image_bytes = BytesIO(contents)

    # Package model, params, and device together
    package = {
        'model': model,
        'params': params,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Call predict_image with correct arguments
    result = predict_image(image_bytes, package)
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_available": len(params['classes']) if params else 0
    }


@app.post("/plant-care")
async def generate_plant_care(file: UploadFile = File(...)):
    """
    Upload an image to get plant classification AND detailed plant care cards from all three agent methods.
    This endpoint combines classification with LLM-powered care instructions using:
    - Direct LLM generation
    - Web search enhanced generation
    - Combined approach merging both methods
    """
    if model is None or params is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Please set environment variable."
        )

    try:
        # Read file contents
        contents = await file.read()
        image_bytes = BytesIO(contents)

        # Package model, params, and device together
        package = {
            'model': model,
            'params': params,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

        # Get plant classification
        prediction = predict_image(image_bytes, package)
        predicted_plant_name = prediction['predicted_class']

        print(f"\n{'='*60}")
        print(f"Predicted plant: {predicted_plant_name}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"{'='*60}\n")

        # Generate plant care cards using all three methods
        print("Generating care card with DIRECT LLM method...")
        care_card_direct = generate_plant_care_card_direct(predicted_plant_name)
        
        print("\nGenerating care card with WEB SEARCH method...")
        care_card_web = generate_plant_care_card_web(predicted_plant_name)
        
        print("\nGenerating care card with COMBINED method...")
        care_card_combined = generate_plant_care_card_combined(predicted_plant_name)

        print("\nAll plant care cards generated successfully!\n")

        # Return classification + all three care cards
        return {
            "classification": prediction,
            "plant_care_cards": {
                "direct_llm": care_card_direct.model_dump(),
                "web_search": care_card_web.model_dump(),
                "combined": care_card_combined.model_dump()
            }
        }

    except ValueError as e:
        # Handle API key or configuration errors
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Handle other errors
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plant care: {str(e)}")


@app.post("/plant-care-rag")
async def generate_plant_care_with_rag(file: UploadFile = File(...)):
    """
    Upload an image to get plant classification AND a detailed plant care card using RAG.
    This endpoint uses Retrieval Augmented Generation (RAG) - it retrieves relevant 
    context from a vector store of plant care guides and uses it to enhance the 
    LLM-generated care card.
    
    The RAG approach provides more accurate and grounded information by:
    - Querying a vector database of verified plant care guides
    - Including retrieved context in the prompt
    - Combining LLM knowledge with specific documented care instructions
    """
    if model is None or params is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if vector_store_manager is None:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. RAG functionality unavailable."
        )

    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Please set environment variable."
        )

    try:
        # Read file contents
        contents = await file.read()
        image_bytes = BytesIO(contents)

        # Package model, params, and device together
        package = {
            'model': model,
            'params': params,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }

        # Get plant classification
        prediction = predict_image(image_bytes, package)
        predicted_plant_name = prediction['predicted_class']

        print(f"\n{'='*60}")
        print(f"Predicted plant: {predicted_plant_name}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"{'='*60}\n")

        # Generate plant care card using RAG
        print("Generating care card with RAG method...")
        care_card_rag = generate_plant_care_card_rag(predicted_plant_name)

        print("\nPlant care card with RAG generated successfully!\n")

        # Return classification + RAG-enhanced care card
        return {
            "classification": prediction,
            "plant_care_card": care_card_rag.model_dump(),
            "method": "RAG (Retrieval Augmented Generation)"
        }

    except ValueError as e:
        # Handle API key or configuration errors
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Handle other errors
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating plant care with RAG: {str(e)}")