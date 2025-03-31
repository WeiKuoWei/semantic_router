import os
import asyncio
import argparse
import time
from pathlib import Path
from utils.converter import CentroidConverter
from utils.visualizer import CentroidVisualizer
from router.multi_layer_router import MultiLayerRouter

import logging
# Set higher logging level for these libraries
logging.getLogger('pikepdf._core').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TRACKING_FILE = os.path.join(BASE_DIR, "tracking", "processed_files.json")
CENTROID_VECTORS_FILE = os.path.join(BASE_DIR, "router", "centroid_vectors.py")

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "tracking"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "router"), exist_ok=True)


async def process_samples():
    """Process and test the semantic router with sample queries."""
    # Create a sample response function
    async def sample_response(query):
        return {"answer": f"I am the expert that can help with: {query}"}
    
    # Initialize router
    router = MultiLayerRouter(use_openai=False)
    
    # Register sample response functions for all experts
    try:
        from router.centroid_vectors import EXPERT_CENTROIDS
        for expert_name in EXPERT_CENTROIDS:
            router.register_expert_response(expert_name, sample_response)
    except ImportError:
        print("No experts found. Please process documents first.")
        return
    
    # Test with some sample queries
    sample_queries = [
        "Tell me about course materials for biology",
        "I need help with my physics homework",
        "I'm feeling stressed about my exams",
        "What are the symptoms of anxiety?",
    ]
    
    for query in sample_queries:
        start_time = time.time()
        print(f"\nRouting query: '{query}'")
        response = await router.route_query(query)
        print(f"Response: {response}, took {time.time() - start_time:.2f} seconds")

async def interactive_mode():
    """Run an interactive session where the user can enter queries."""
    # Create a sample response function
    async def sample_response(query):
        return {"answer": f"I am the expert that can help with: {query}"}
    
    # Initialize router
    print("Initializing router...")
    router = MultiLayerRouter(use_openai=False)
    
    # Register sample response functions for all experts
    try:
        from router.centroid_vectors import EXPERT_CENTROIDS, EXPERT_TO_GROUP
        for expert_name in EXPERT_CENTROIDS:
            router.register_expert_response(expert_name, sample_response)
        
        print(f"Found {len(EXPERT_CENTROIDS)} experts across multiple groups.")
    except ImportError:
        print("No experts found. Please process documents first.")
        return
    
    print("\nEnter your queries (type 'exit', 'quit', or 'q' to quit):")
    
    while True:
        # Get query from user
        query = input("\n> ")
        
        # Check if user wants to exit
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting interactive mode.")
            break
        
        # Process the query
        if query.strip():
            start_time = time.time()
            try:
                response = await router.route_query(query)
                print(f"Response: {response['answer']}")
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing query: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Semantic Router")
    parser.add_argument("--process", action="store_true", help="Process documents and update centroids")
    parser.add_argument("--test", action="store_true", help="Test the router with sample queries")
    parser.add_argument("--query", action="store_true", help="Enter interactive mode to try custom queries")
    args = parser.parse_args()
    
    if args.process:
        print("Processing documents and updating centroids...")
        
        # Process documents and update centroids
        converter = CentroidConverter(DATA_DIR, TRACKING_FILE)
        tracking_data = converter.process_all()
        
        # Generate centroid vectors file
        visualizer = CentroidVisualizer(TRACKING_FILE, CENTROID_VECTORS_FILE)
        visualizer.generate_centroid_vectors_file()
        
        print("Processing complete.")
    
    if args.test:
        print("Testing router with sample queries...")
        asyncio.run(process_samples())
    
    if args.query:
        print("Entering interactive mode...")
        asyncio.run(interactive_mode())
    
    if not (args.process or args.test or args.query):
        print("Please specify an action: --process, --test, --try, or --query")
        print("Example: python main.py --process")
        print("Example: python main.py --test")
        print("Example: python main.py --query")


if __name__ == "__main__":
    main()