"""Response schema for plant care agent."""

from pydantic import BaseModel, Field
from .plant_care_card import PlantCareCard


class PlantCareResponse(BaseModel):
    """Response schema for the Plant Care agent, containing a PlantCareCard."""
    plant_care_card: PlantCareCard = Field(description="A detailed plant care data card for the requested plant.")
