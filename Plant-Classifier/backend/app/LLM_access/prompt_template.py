"""Prompt templates for plant care generation."""

from langchain_core.prompts import ChatPromptTemplate


plant_care_prompt_template = ChatPromptTemplate([
    ("system", """You are a world-class botanist and an expert in plant care.
      Your task is to provide comprehensive and accurate plant care information in a structured format.
      You have decades of experience with both common houseplants and rare species."""),
    ("user", "Generate a detailed plant care data card. Your output MUST be a JSON object following the structure of the PlantCareCard."),
    ("user", "I need a detailed plant care data card for a specific plant."),
    ("user", "Make sure to include specific, actionable advice based on the latest horticultural research."),
    ("user", "Now, generate the Plant Care Card for {plant_name}:")
])
