import os
import asyncio
import argparse
import time
import uuid
import logging
from pathlib import Path
from utils.converter import CentroidConverter
from utils.visualizer import CentroidVisualizer
from router.multi_layer_router import MultiLayerRouter
from utils.session_manager import SessionManager
from utils.config import BASE_DIR, SRC_DIR, DATA_DIR

# Set higher logging level for these libraries
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('chromadb').setLevel(logging.WARNING)
logging.getLogger('pikepdf._core').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define additional paths
TRACKING_FILE = os.path.join(SRC_DIR, "tracking", "processed_files.json")
CENTROID_VECTORS_FILE = os.path.join(SRC_DIR, "router", "centroid_vectors.py")
CONVERSATION_DIR = os.path.join(SRC_DIR, "tracking", "conversations")

# Ensure directories exist
os.makedirs(os.path.join(SRC_DIR, "tracking"), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, "router"), exist_ok=True)
os.makedirs(CONVERSATION_DIR, exist_ok=True)


async def process_samples():
    """Process and test the semantic router with sample queries."""
    # Create session manager
    session_manager = SessionManager(os.path.join(SRC_DIR, "tracking"))
    
    # Initialize router with session manager
    router = MultiLayerRouter(use_openai=False, session_manager=session_manager)
    
    # Generate a sample user ID
    user_id = f"test_user_default"
    
    # Test with some sample queries
    sample_queries = [
        "Tell me about course materials for biology",
        "I need help with my physics homework",
        "I'm feeling stressed about my exams",
        "What are the symptoms of anxiety?",
        "new_session",  # Test session reset
    ]
    
    for query in sample_queries:
        print(f"\nRouting query: '{query}'")
        start_time = time.time()
        response = await router.route_query(query, user_id)
        print(f"Response: {response}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

async def interactive_mode():
    """Run an interactive session where the user can enter queries."""
    # Create session manager
    session_manager = SessionManager(os.path.join(SRC_DIR, "tracking"))
    
    # Initialize router
    print("Initializing router...")
    router = MultiLayerRouter(use_openai=False, session_manager=session_manager)
    
    # Generate a user ID for this session
    user_id = f"interactive_user_default"
    print(f"Session ID: {user_id}")
    
    # Check if experts exist
    try:
        from router.centroid_vectors import EXPERT_CENTROIDS, EXPERT_TO_GROUP
        print(f"Found {len(EXPERT_CENTROIDS)} experts across multiple groups.")
    except ImportError:
        print("No experts found. Please process documents first.")
        return
    
    print("\nEnter your queries (type 'exit', 'quit', or 'q' to quit)")
    print("Type 'new_session', 'reset', or 'clear' to start a new conversation")
    
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
                response = await router.route_query(query, user_id)
                
                # Pretty print the response
                print("\n" + "="*80)
                print(" RESPONSE ".center(80, "="))
                print("="*80 + "\n")
                
                # Print answer with word wrapping
                import textwrap
                answer = response.get('answer', 'No answer provided')
                wrapped_answer = textwrap.fill(answer, width=80)
                print(wrapped_answer)
                
                # Print sources info if available
                if 'sources' in response:
                    print("\n" + "-"*80)
                    print(f"Sources: {response['sources']} documents retrieved")
                
                print("\n" + "="*80)
                
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
        tracking_data = asyncio.run(converter.process_all())
        
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
        print("Please specify an action: --process, --test, or --query")
        print("Example: python src/main.py --process")
        print("Example: python src/main.py --test")
        print("Example: python src/main.py --query")


if __name__ == "__main__":
    main()