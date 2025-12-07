"""Vector store management using LanceDB."""

import os
from typing import List
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages LanceDB vector store for plant care documents."""
    
    def __init__(self, db_path: str = "resources/vector_store"):
        """
        Initialize vector store manager.
        
        Args:
            db_path: Path to LanceDB database directory
        """
        self.db_path = db_path
        self.connection = None
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=300,
            chunk_overlap=25
        )
    
    def initialize(self):
        """Initialize or connect to existing vector store."""
        self.connection = lancedb.connect(self.db_path)
        
        # Try to load existing vector store
        try:
            self.vector_store = LanceDB(
                connection=self.connection,
                embedding=self.embeddings
            )
            print(f"Connected to existing vector store at {self.db_path}")
        except Exception as e:
            print(f"No existing vector store found. Will create new one. Error: {e}")
    
    def load_documents_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Load documents from URLs.
        
        Args:
            urls: List of URLs to load
            
        Returns:
            List of loaded Document objects
        """
        print(f"Loading documents from {len(urls)} URLs...")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        print(f"Splitting {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def build_vector_store(self, documents: List[Document]):
        """
        Build vector store from documents.
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("No documents provided to build vector store")
        
        print(f"Building vector store with {len(documents)} documents...")
        self.vector_store = LanceDB.from_documents(
            documents=documents,
            embedding=self.embeddings,
            connection=self.connection
        )
        print("Vector store built successfully")
    
    def query(self, query: str, k: int = 5) -> List[Document]:
        """
        Query the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() or build_vector_store() first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def query_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Query the vector store with relevance scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() or build_vector_store() first.")
        
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return results


# Global vector store instance (will be initialized on startup)
_vector_store_manager: VectorStoreManager = None


def get_vector_store_manager() -> VectorStoreManager:
    """Get the global vector store manager instance."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
        _vector_store_manager.initialize()
    return _vector_store_manager
