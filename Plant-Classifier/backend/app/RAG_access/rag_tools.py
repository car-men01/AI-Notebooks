"""RAG tools for retrieving plant care context from vector store."""

from langchain.tools import tool as langchain_tool

# Global reference to vector store manager (set by main.py)
_vector_store_manager = None

def set_vector_store_manager(manager):
    """Set the global vector store manager instance."""
    global _vector_store_manager
    _vector_store_manager = manager


@langchain_tool
def retrieve_plant_care_context(plant_name: str) -> str:
    """
    Retrieves relevant plant care context from the LanceDB vector store for a given plant name.
    
    Args:
        plant_name: Name of the plant to retrieve care information for
        
    Returns:
        Concatenated content from relevant document chunks
    """
    print(f"Retrieving context for {plant_name} from vector store...")
    
    if _vector_store_manager is None:
        print("Warning: Vector store manager not initialized")
        return "Vector store not available. Please initialize the vector store first."
    
    try:
        query = f"care guide for {plant_name}"
        docs = _vector_store_manager.query(query, k=5)
        
        if not docs:
            print(f"No relevant documents found for {plant_name}")
            return f"No specific care information found for {plant_name} in the vector store."
        
        context_content = "---\n".join([doc.page_content for doc in docs])
        print(f"Found {len(docs)} relevant document chunks for {plant_name}.")
        return context_content
    
    except Exception as e:
        print(f"Error retrieving context for {plant_name}: {e}")
        return f"Error retrieving information: {str(e)}"
