"""Utility script to populate the vector store with plant care documents.

Run this script to build the vector store from web-based plant care guides.
You can modify the plant_urls dictionary to add or change plant care sources.
"""

import os
import sys

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.RAG_access.vector_store import VectorStoreManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dictionary of plant names and their care guide URLs
# You can add more plants and URLs here
plant_urls = {
    "aloe vera": [
        "https://www.thespruce.com/grow-aloe-vera-4775803",
    ],
    "banana": [
        "https://www.gardeningknowhow.com/edible/fruits/banana/growing-banana-trees.htm",
    ],
    "coconut": [
        "https://www.gardeningknowhow.com/edible/fruits/coconut/growing-coconut-palms.htm",
    ],
    "kale": [
        "https://www.almanac.com/plant/kale",
    ],
    "papaya": [
        "https://www.gardeningknowhow.com/edible/fruits/papaya/growing-papayas.htm",
    ],
    "pineapple": [
        "https://www.thespruce.com/how-to-grow-pineapple-plants-4125483",
    ],
}


def main():
    """Main function to populate vector store."""
    print("=" * 60)
    print("Vector Store Population Script")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        print("Please set it in your .env file or environment variables")
        return
    
    # Initialize vector store manager
    print("\nInitializing vector store manager...")
    vs_manager = VectorStoreManager()
    
    # Collect all URLs
    all_urls = []
    for plant, urls in plant_urls.items():
        print(f"  - {plant}: {len(urls)} URL(s)")
        all_urls.extend(urls)
    
    print(f"\nTotal URLs to process: {len(all_urls)}")
    
    # Load documents from URLs
    print("\nLoading documents from URLs...")
    try:
        documents = vs_manager.load_documents_from_urls(all_urls)
        print(f"Successfully loaded {len(documents)} documents")
    except Exception as e:
        print(f"ERROR loading documents: {e}")
        return
    
    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    try:
        chunks = vs_manager.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
    except Exception as e:
        print(f"ERROR splitting documents: {e}")
        return
    
    # Build vector store
    print("\nBuilding vector store...")
    try:
        vs_manager.build_vector_store(chunks)
        print("Vector store built successfully!")
    except Exception as e:
        print(f"ERROR building vector store: {e}")
        return
    
    # Test the vector store
    print("\nTesting vector store with a sample query...")
    test_query = "How to care for aloe vera?"
    try:
        results = vs_manager.query(test_query, k=3)
        print(f"Found {len(results)} results for: '{test_query}'")
        if results:
            print(f"\nFirst result preview (first 200 chars):")
            print(results[0].page_content[:200] + "...")
    except Exception as e:
        print(f"ERROR querying vector store: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Vector store population completed successfully!")
    print("=" * 60)
    print(f"\nVector store location: {vs_manager.db_path}")
    print("You can now use the /plant-care-rag endpoint in your API")


if __name__ == "__main__":
    main()
