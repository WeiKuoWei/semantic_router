import os
import asyncio
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers import SentenceTransformer

from utils.config import DB_PATH


class ChromaDBHandler:
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize the ChromaDBHandler.
        
        Args:
            db_path: Path to the ChromaDB database
        """
        self.db_path = Path(db_path)
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Ensure database directory exists
        os.makedirs(self.db_path, exist_ok=True)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text document.
        
        Args:
            text: The document text to embed
            
        Returns:
            The embedding vector as a list
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    async def save_to_db(self, collection_name: str, docs: List[Document], embeddings: List[List[float]], metadatas: List[Dict]):
        """
        Save documents to ChromaDB.
        
        Args:
            collection_name: Name of the collection
            docs: List of documents
            embeddings: List of embeddings
            metadatas: List of metadata dictionaries
        """
        # Create or get collection
        collection = self.client.get_or_create_collection(collection_name)
        
        # Generate IDs if not in metadata
        ids = [str(metadata.get('id', uuid.uuid4())) for metadata in metadatas]
        
        # Extract text content
        documents = [doc.page_content for doc in docs]
        
        # Upsert to database
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Saved {len(docs)} documents to collection '{collection_name}'")
    
    async def get_similar_documents(self, collection_name: str, query_embedding: List[float], top_k: int = 3) -> Dict:
        """
        Get similar documents from ChromaDB.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Embedding of the query
            top_k: Number of results to return
            
        Returns:
            Dictionary of results
        """
        try:
            # Get collection
            collection = self.client.get_collection(collection_name)
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    async def process_and_save_pdfs(self, expert_dir: Path, collection_name: str) -> Set[str]:
        """
        Process PDFs from an expert directory and save them to ChromaDB.
        
        Args:
            expert_dir: Path to the expert directory
            collection_name: Name of the collection
            
        Returns:
            Set of processed file names
        """
        # Load PDFs
        try:
            loader = DirectoryLoader(str(expert_dir), glob="*.pdf", show_progress=True)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {expert_dir}")
        except Exception as e:
            print(f"Error loading documents from {expert_dir}: {e}")
            return set()
        
        if not docs:
            return set()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # Get filenames
        filenames = {Path(doc.metadata['source']).name for doc in docs}
        
        # Calculate embeddings and prepare metadata
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            embedding = self.get_embedding(chunk.page_content)
            embeddings.append(embedding)
            
            # Create metadata
            metadata = {
                'id': str(uuid.uuid4()),
                'source': Path(chunk.metadata['source']).name,
                'expert': collection_name
            }
            metadatas.append(metadata)
        
        # Save to database
        await self.save_to_db(collection_name, chunks, embeddings, metadatas)
        
        return filenames