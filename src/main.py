import os
import asyncio
import argparse
import time
import logging
from pathlib import Path
from utils.converter import CentroidConverter
from utils.visualizer import CentroidVisualizer
from router.multi_layer_router import MultiLayerRouter
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

# Ensure directories exist
os.makedirs(os.path.join(SRC_DIR, "tracking"), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, "router"), exist_ok=True)


async def process_samples():
    """Process and test the semantic router with sample queries."""
    # Import the samples
    from expert_samples import samples
    
    # Initialize router
    router = MultiLayerRouter(use_openai=False)
    
    # Track accuracy metrics per expert
    results = {}
    overall_correct = 0
    overall_total = 0
    total_time = 0

    # Process each expert's samples
    for expert, queries in samples.items():
        print(f"\n===== Testing expert: {expert} =====")
        
        # Initialize metrics for this expert
        results[expert] = {
            "total": len(queries),
            "correct": 0,
            "incorrect_routings": {}
        }
        
        # Process each query for this expert
        for query in queries:
            print(f"\nRouting query: '{query}'")
            print(f"Expected expert: {expert}")
            
            start_time = time.time()
            _, response = await router.route_query(
                query=query, 
                check_accuracy=True, 
                expected_expert=expert
            )
            
            # Update accuracy counters
            is_correct = response.get("is_correct", False)
            if is_correct:
                results[expert]["correct"] += 1
                overall_correct += 1
            else:
                # Track which expert it was incorrectly routed to
                routed_expert = response.get("expert", "unknown")
                if routed_expert not in results[expert]["incorrect_routings"]:
                    results[expert]["incorrect_routings"][routed_expert] = 0
                results[expert]["incorrect_routings"][routed_expert] += 1
            
            overall_total += 1
            time_taken = time.time() - start_time
            total_time += time_taken
            print(f"Time taken: {time_taken:.2f} seconds")
    
    # Print overall accuracy statistics
    if overall_total > 0:
        print("\n===== Per-Expert Accuracy =====")
        for expert, metrics in results.items():
            accuracy = (metrics["correct"] / metrics["total"]) * 100
            print(f"\nExpert: {expert}")
            print(f"Accuracy: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
            
            if metrics["incorrect_routings"]:
                print("Incorrect routings:")
                for wrong_expert, count in metrics["incorrect_routings"].items():
                    print(f"  - {wrong_expert}: {count} times")

        print("\n===== Overall Accuracy Results =====")
        print(f"Total samples: {overall_total}")
        print(f"Correctly routed: {overall_correct}")
        print(f"Overall accuracy: {(overall_correct / overall_total) * 100:.2f}%")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / overall_total:.3f} seconds")
    else:
        print("No samples were processed.")

async def interactive_mode():
    """Run an interactive session where the user can enter queries."""
    # Initialize router
    print("Initializing router...")
    router = MultiLayerRouter(use_openai=False)
    
    # Check if experts exist
    try:
        from router.centroid_vectors import EXPERT_CENTROIDS, EXPERT_TO_GROUP
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
                expert, _ = await router.route_query(query)
                print(f"Best expert: {expert}")
                # print(response)
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
        _ = asyncio.run(converter.process_all())
        
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