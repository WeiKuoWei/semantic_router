import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

class CentroidConverter:
    def __init__(self, base_dir: str, tracking_file: str):
        """
        Initialize the CentroidConverter.
        
        Args:
            base_dir: The base directory containing group folders
            tracking_file: Path to the JSON file that tracks processed files
        """
        self.base_dir = Path(base_dir)
        self.tracking_file = Path(tracking_file)
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load tracking data if it exists
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Check if file is not empty
                        self.tracking_data = json.load(f)
                    else:
                        self.tracking_data = {}
            except json.JSONDecodeError:
                # If the file exists but has invalid JSON
                print(f"Warning: Invalid JSON in tracking file. Creating new tracking data.")
                self.tracking_data = {}
        else:
            self.tracking_data = {}
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
    
    def save_tracking_data(self):
        """Save the current tracking data to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text document.
        
        Args:
            text: The document text to embed
            
        Returns:
            The embedding vector as numpy array
        """
        return self.model.encode(text)
    
    def calculate_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate the centroid of a list of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            The centroid vector
        """
        if not embeddings:
            raise ValueError("Cannot calculate centroid of empty embeddings list")
        
        # Stack and calculate mean
        embeddings_array = np.vstack(embeddings)
        return np.mean(embeddings_array, axis=0)
    
    def get_expert_data(self, expert_dir: Path) -> Tuple[np.ndarray, Set[str]]:
        """
        Process an expert directory and get its centroid.
        
        Args:
            expert_dir: Path to the expert directory containing PDFs
            
        Returns:
            Tuple of (centroid_vector, processed_files)
        """
        expert_name = expert_dir.name
        group_name = expert_dir.parent.name
        
        # Get tracked data for this expert
        if group_name not in self.tracking_data:
            self.tracking_data[group_name] = {"centroid": None, "experts": {}}
            
        group_data = self.tracking_data[group_name]
        
        if expert_name not in group_data["experts"]:
            group_data["experts"][expert_name] = {"centroid": None, "files": []}
            
        expert_data = group_data["experts"][expert_name]
        tracked_files = set(expert_data["files"])
        
        # Load all PDFs from this directory
        loader = DirectoryLoader(str(expert_dir), glob="*.pdf", show_progress=True)
        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {expert_dir}")
        except Exception as e:
            print(f"Error loading documents from {expert_dir}: {e}")
            return None, tracked_files
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        
        # Get filenames of all documents
        current_files = {doc.metadata['source'].split('/')[-1] for doc in docs}
        
        # Only process new files
        new_files = current_files - tracked_files
        
        if not new_files:
            print(f"No new files to process in {expert_dir}")
            # Return existing centroid if available
            if expert_data["centroid"] is not None:
                return np.array(expert_data["centroid"]), tracked_files
            else:
                # Process all files if centroid doesn't exist
                new_files = current_files
        
        # Calculate embeddings
        all_embeddings = []
        
        # If we have an existing centroid, include it weighted by the number of files
        if expert_data["centroid"] is not None and tracked_files:
            existing_centroid = np.array(expert_data["centroid"])
            # Weight the existing centroid by the number of tracked files
            all_embeddings.append(existing_centroid * len(tracked_files))
            weight_sum = len(tracked_files)
        else:
            weight_sum = 0
        
        # Process new chunks
        for chunk in chunks:
            # Only process chunks from new files
            source_file = chunk.metadata['source'].split('/')[-1]
            if source_file in new_files:
                embedding = self.get_document_embedding(chunk.page_content)
                all_embeddings.append(embedding)
                weight_sum += 1
        
        # Calculate new centroid
        if all_embeddings:
            combined_vector = np.sum(all_embeddings, axis=0)
            centroid = combined_vector / weight_sum
            
            # Update tracking data
            expert_data["centroid"] = centroid.tolist()
            expert_data["files"] = list(tracked_files.union(new_files))
            
            return centroid, tracked_files.union(new_files)
        
        return np.array(expert_data["centroid"]), tracked_files
    
    def process_all(self) -> Dict:
        """
        Process all groups and experts, updating centroids.
        
        Returns:
            The updated tracking data
        """
        # Loop through all directories in the base directory (groups)
        for group_dir in self.base_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            group_name = group_dir.name
            print(f"Processing group: {group_name}")
            
            expert_centroids = []
            expert_weights = []
            
            # Process each expert directory
            for expert_dir in group_dir.iterdir():
                if not expert_dir.is_dir():
                    continue
                    
                expert_name = expert_dir.name
                print(f"Processing expert: {expert_name}")
                
                centroid, files = self.get_expert_data(expert_dir)
                if centroid is not None:
                    expert_centroids.append(centroid)
                    expert_weights.append(len(files))
            
            # Calculate group centroid
            if expert_centroids:
                # Weight by number of files
                weighted_centroids = [c * w for c, w in zip(expert_centroids, expert_weights)]
                group_centroid = np.sum(weighted_centroids, axis=0) / sum(expert_weights)
                self.tracking_data[group_name]["centroid"] = group_centroid.tolist()
        
        # Save the updated tracking data
        self.save_tracking_data()
        return self.tracking_data