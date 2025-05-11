from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, sys, logging

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.multi_layer_router import MultiLayerRouter
from utils.session_manager import SessionManager
from utils.converter import CentroidConverter
from utils.visualizer import CentroidVisualizer
from utils.config import SRC_DIR, DATA_DIR, TRACKING_FILE, CENTROID_VECTORS_FILE
from api.models.schemas import QueryRequest, QueryResponse, HistoryResponse

# First configure the basic logging format
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s") # "%(asctime)s 

# Then configure specific loggers
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)
logging.getLogger('pikepdf._core').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfpage').setLevel(logging.ERROR)
logging.getLogger("llm_service").setLevel(logging.INFO)
# logging.getLogger("multi_layer_router").setLevel(logging.INFO)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(
    title="Expert AI Router API", 
    docs_url="/docs", 
)

# Setup CORS to allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
session_manager = SessionManager("tracking")
router = MultiLayerRouter(use_openai=False, session_manager=session_manager)

@app.on_event("shutdown")
async def shutdown_event():
    # Close any open resources safely
    if hasattr(router, "db_handler"):
        try:
            router.db_handler.close()
        except AttributeError:
            # Log that closing failed but don't crash
            print("Warning: Could not close ChromaDB client properly")

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return response from appropriate expert"""
    # Generate session ID if not provided
    session_id = request.session_id or f"user_default"
    print(f"Session ID: {session_id}")

    # Process query through router
    response, best_expert = await router.route_query(request.query, session_id)
    
    # Process the best_expert format: break down by "_" and capitalize
    if best_expert:
        best_expert = best_expert.replace("_", " ").title()
    else:
        best_expert = "unknown"

    return {
        "answer": response.get("answer", "No response generated"),
        "session_id": session_id,
        "expert": best_expert,
        "sources": response.get("sources", 0)
    }

@app.post("/api/process_file")
async def process_file():
    # Process documents and update centroids
    converter = CentroidConverter(DATA_DIR, TRACKING_FILE)
    await converter.process_all()
    
    # Generate centroid vectors file
    visualizer = CentroidVisualizer(
        tracking_file=TRACKING_FILE, output_file=CENTROID_VECTORS_FILE
        )
    visualizer.generate_centroid_vectors_file()

@app.get("/api/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get conversation history for a session"""
    history = session_manager.get_history(session_id)
    return {"history": history}

@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session"""
    session_manager.clear_session(session_id)
    return {"status": "success", "message": "Session cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)