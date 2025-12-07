# API request/response models

from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3_predictions: List[dict]