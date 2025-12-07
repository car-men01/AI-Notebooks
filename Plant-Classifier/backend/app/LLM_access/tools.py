"""Tools for plant care information retrieval."""

from langchain.tools import tool as langchain_tool
from tavily import TavilyClient
from loguru import logger

from .prompt_template import plant_care_prompt_template
from ..config import get_tavily_client


@langchain_tool
def plant_care_tool_simple(plant_name: str, model) -> str:
    """
    Retrieves detailed plant care information in a text format for a given plant name.
    The agent is expected to parse this text into a structured format if needed.
    """
    print(f"Tool call: Retrieving plant care information in text for: {plant_name}...")

    # Invoke the prompt with the plant name to get the chat messages
    messages = plant_care_prompt_template.invoke({"plant_name": plant_name})

    # Use the base LLM to get a raw text response
    raw_text_response = model.invoke(messages).content

    print(f"Tool executed successfully for {plant_name}, returning raw text.")
    return raw_text_response


@langchain_tool
def plant_care_tool(plant_name: str) -> str:
    """
    Searches the web for plant care information using Tavily API.

    Args:
        plant_name (str): The name of the plant to get care instructions for.
    Returns:
        str: Raw web search results about the plant care.
    """

    if not plant_name:
        raise ValueError("plant_name is required")

    # Get Tavily client from config
    tavily_client = get_tavily_client()

    # Search the web for plant care information using Tavily
    search_query = f"{plant_name} latin plant care watering sunlight propagation requirements tips"

    try:
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5
        )

        # Extract and format search results
        formatted_results = []
        for i, result in enumerate(search_results.get('results', []), 1):
            formatted_results.append(
                f"Result {i}:\n"
                f"Source: {result['url']}\n"
                f"Content: {result['content']}\n"
            )

        search_context = "\n".join(formatted_results)

        logger.info(f"Found {len(search_results.get('results', []))} search results for {plant_name}")

        return search_context if search_context else "No search results found."

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error during web search: {str(e)}"
    
