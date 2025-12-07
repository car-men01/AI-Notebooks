"""Functions for generating plant care cards using LLM agents."""

from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from .plant_care_card import PlantCareCard, LightingConditions, Watering, TemperatureRange, Humidity
from .plant_care_response import PlantCareResponse
from .prompt_template import plant_care_prompt_template
from .tools import plant_care_tool, plant_care_tool_simple
from ..config import get_openai_model


def generate_plant_care_card_direct(plant_name: str) -> PlantCareCard:
    """
    Generates a structured PlantCareCard directly from the LLM for a given plant name,
    enforcing the Pydantic schema.
    """

    print(f"Generating Plant Care Card for: {plant_name}...")

    # Get the model from config
    model = get_openai_model()

    # Create a new model instance that is configured to return structured output
    # directly as a PlantCareCard Pydantic object.
    structured_model = model.with_structured_output(PlantCareCard)

    # Invoke the structured model with the prompt. The output will automatically
    # be a validated PlantCareCard Pydantic object.
    plant_care_card = structured_model.invoke(plant_care_prompt_template.invoke({"plant_name": plant_name}))

    print(f"Plant Care Card generated and parsed successfully for {plant_name}.")
    return plant_care_card

def generate_plant_care_card_web(plant_name: str) -> PlantCareCard:
    """
    Generates a structured PlantCareCard using the agent with web search for a given plant name.
    """
    print(f"Generating Plant Care Card with web search for: {plant_name}...")

    # Get the model from config
    model = get_openai_model()

    # Create agent prompt
    agent_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(plant_care_prompt_template.messages[0].prompt.template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(plant_care_prompt_template.messages[-1].prompt.template)
    ])

    # Create agent with web search tool
    agent = create_react_agent(
        model,
        tools=[plant_care_tool],
        response_format=PlantCareResponse,
    )

    # Create the prompt for the agent
    prompt_with_plant_name = agent_prompt.invoke({"plant_name": plant_name, "chat_history": []})

    # Invoke the agent with web search access
    response = agent.invoke(prompt_with_plant_name)

    # Extract the PlantCareCard object from the agent's structured response
    plant_care_card = response['structured_response'].plant_care_card

    print(f"Plant Care Card generated and parsed successfully with web search for {plant_name}.")
    return plant_care_card

def merge_plant_care_cards(card1: PlantCareCard, card2: PlantCareCard) -> PlantCareCard:
    """
    Intelligently merges two PlantCareCard instances into a single, more comprehensive one.
    Prioritizes card2 (web search) for most fields, and combines notes/ranges where appropriate.
    """

    merged_data = {}

    # Helper function to merge notes
    def merge_notes(note1: Optional[str], note2: Optional[str]) -> Optional[str]:
        if note1 and note2 and note1 != note2:
            return f"{note1}; {note2}"
        return note2 or note1

    # Simple string fields (plant_name, latin_name, soil_type, propagation, special_care)
    # Prioritize card2, if not available use card1
    merged_data['plant_name'] = card2.plant_name if card2.plant_name else card1.plant_name
    merged_data['latin_name'] = card2.latin_name if card2.latin_name else card1.latin_name
    merged_data['soil_type'] = card2.soil_type if card2.soil_type else card1.soil_type
    merged_data['propagation'] = card2.propagation if card2.propagation else card1.propagation
    merged_data['special_care'] = card2.special_care if card2.special_care else card1.special_care

    # Boolean field (outdoors)
    # If either is True, result is True. Prioritize card2 if card1 is None/False.
    merged_data['outdoors'] = card2.outdoors or card1.outdoors

    # Nested Pydantic Models

    # LightingConditions
    merged_lighting = LightingConditions(
        type=card2.lighting_conditions.type or card1.lighting_conditions.type,
        duration=card2.lighting_conditions.duration or card1.lighting_conditions.duration,
        notes=merge_notes(card1.lighting_conditions.notes, card2.lighting_conditions.notes)
    )
    merged_data['lighting_conditions'] = merged_lighting

    # Watering
    merged_watering = Watering(
        frequency=card2.watering.frequency or card1.watering.frequency,
        method=card2.watering.method or card1.watering.method,
        notes=merge_notes(card1.watering.notes, card2.watering.notes)
    )
    merged_data['watering'] = merged_watering

    # TemperatureRange - apply 'most inclusive range' logic
    min_celsius = card1.temperature_range.min_celsius
    if card2.temperature_range.min_celsius is not None:
        if min_celsius is not None:
            min_celsius = min(min_celsius, card2.temperature_range.min_celsius)
        else:
            min_celsius = card2.temperature_range.min_celsius

    max_celsius = card1.temperature_range.max_celsius
    if card2.temperature_range.max_celsius is not None:
        if max_celsius is not None:
            max_celsius = max(max_celsius, card2.temperature_range.max_celsius)
        else:
            max_celsius = card2.temperature_range.max_celsius

    merged_temp_range = TemperatureRange(
        min_celsius=min_celsius,
        max_celsius=max_celsius,
        notes=merge_notes(card1.temperature_range.notes, card2.temperature_range.notes)
    )
    merged_data['temperature_range'] = merged_temp_range

    # Humidity
    merged_humidity = Humidity(
        level=card2.humidity.level or card1.humidity.level,
        notes=merge_notes(card1.humidity.notes, card2.humidity.notes)
    )
    merged_data['humidity'] = merged_humidity

    return PlantCareCard(**merged_data)

def generate_plant_care_card_combined(plant_name: str) -> PlantCareCard:
    """
    Generates a comprehensive PlantCareCard by combining results from both
    the direct LLM agent and the web search agent.
    """
    print(f"Generating combined Plant Care Card for: {plant_name}...")

    # Generate card from direct LLM agent
    card_direct = generate_plant_care_card_direct(plant_name)

    # Generate card from web search agent
    card_web = generate_plant_care_card_web(plant_name)

    # Merge the two cards
    combined_card = merge_plant_care_cards(card_direct, card_web)

    print(f"Combined Plant Care Card generated successfully for {plant_name}.")
    return combined_card