"""Pydantic models for plant care card structure."""

from pydantic import BaseModel, Field
from typing import Optional


class LightingConditions(BaseModel):
    """Structured details for a plant's lighting conditions."""
    type: str = Field(description="The type of light the plant needs (e.g., 'Full sun', 'Bright indirect light', 'Low light').")
    duration: Optional[str] = Field(None, description="Optional duration of light exposure (e.g., '6+ hours daily').")
    notes: Optional[str] = Field(None, description="Optional additional notes or tips regarding lighting.")

class Watering(BaseModel):
    """Structured details for a plant's watering needs."""
    frequency: str = Field(description="How often to water the plant (e.g., 'Consistently moist', 'Allow soil to dry between waterings').")
    method: Optional[str] = Field(None, description="Optional method of watering (e.g., 'Water deeply', 'Mist leaves').")
    notes: Optional[str] = Field(None, description="Optional additional notes or tips regarding watering.")

class TemperatureRange(BaseModel):
    """Structured details for a plant's ideal temperature range."""
    min_celsius: Optional[float] = Field(None, description="Minimum ideal temperature in Celsius.")
    max_celsius: Optional[float] = Field(None, description="Maximum ideal temperature in Celsius.")
    notes: Optional[str] = Field(None, description="Additional notes or context about the temperature range.")

class Humidity(BaseModel):
    """Structured details for a plant's ideal humidity."""
    level: str = Field(description="General humidity level (e.g., 'Low', 'Moderate', 'High', 'Very High').")
    notes: Optional[str] = Field(None, description="Additional notes or tips regarding humidity.")

class PlantCareCard(BaseModel):
    """Structured output for a plant care data card."""
    plant_name: str = Field(description="The common name of the plant.")
    latin_name: str = Field(description="The scientific (Latin) name of the plant.")
    outdoors: bool = Field(description="True if the plant is typically grown outdoors, False if indoors.")
    lighting_conditions: LightingConditions = Field(description="Describes the ideal lighting conditions for the plant.")
    watering: Watering = Field(description="Instructions on how frequently and how much to water the plant.")
    soil_type: str = Field(description="Recommended soil type for the plant.")
    temperature_range: TemperatureRange = Field(description="Ideal temperature range for the plant.")
    humidity: Humidity = Field(description="Ideal humidity levels for the plant.")
    propagation: str = Field(description="Common methods for propagating the plant.")
    special_care: str = Field(description="Any additional special care notes, tips, or common problems.")
