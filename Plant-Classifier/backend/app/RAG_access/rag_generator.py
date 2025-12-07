"""RAG-enhanced plant care card generation."""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from ..LLM_access.plant_care_card import PlantCareCard
from ..LLM_access.plant_care_response import PlantCareResponse
from ..LLM_access.tools import plant_care_tool
from ..config import get_openai_model
from .rag_tools import retrieve_plant_care_context


# RAG-enhanced prompt template with context
plant_care_prompt_template_rag = ChatPromptTemplate([
    ("system", """You are a world-class botanist and an expert in plant care.
      Your task is to provide comprehensive and accurate plant care information in a structured format.
      You have decades of experience with both common houseplants and rare species."""),
    ("user", "Generate a detailed plant care data card. Your output MUST be a JSON object following the structure of the PlantCareCard."),
    ("user", "I need a detailed plant care data card for a specific plant."),
    ("user", "Make sure to include specific, actionable advice based on the latest horticultural research."),
    ("user", "Here is some additional retrieved context from verified care guides that might be useful:\n\n{context}\n"),
    ("user", "Now, generate the Plant Care Card for {plant_name}:")
])


def generate_plant_care_card_rag(plant_name: str) -> PlantCareCard:
    """
    Generates a structured PlantCareCard using RAG - retrieves context from vector store
    and uses it to enhance the LLM generation.
    
    Args:
        plant_name: Name of the plant to generate care card for
        
    Returns:
        PlantCareCard object with care information
    """
    print(f"Generating Plant Care Card with RAG for: {plant_name}...")

    # Get the model from config
    model = get_openai_model()

    # Retrieve context from the vector store using the tool
    retrieved_context = retrieve_plant_care_context.invoke({"plant_name": plant_name})
    
    # Create agent prompt with context
    agent_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(plant_care_prompt_template_rag.messages[0].prompt.template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(plant_care_prompt_template_rag.messages[-1].prompt.template)
    ])

    # Create agent with web search tool
    agent = create_react_agent(
        model,
        tools=[plant_care_tool],
        response_format=PlantCareResponse,
    )

    # Create the prompt for the agent, including the retrieved context
    prompt_with_plant_name_and_context = agent_prompt.invoke({
        "plant_name": plant_name,
        "chat_history": [],
        "context": retrieved_context
    })

    # Invoke the agent
    response = agent.invoke(prompt_with_plant_name_and_context)

    # Extract the PlantCareCard object from the agent's structured response
    plant_care_card = response['structured_response'].plant_care_card

    print(f"Plant Care Card generated successfully with RAG for {plant_name}.")
    return plant_care_card
