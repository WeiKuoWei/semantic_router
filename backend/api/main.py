from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import sys

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.multi_layer_router import MultiLayerRouter
from utils.session_manager import SessionManager
from api.models.schemas import QueryRequest, QueryResponse, HistoryResponse

app = FastAPI(title="Expert AI Router API")

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

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query and return response from appropriate expert"""
    # Generate session ID if not provided
    session_id = request.session_id or f"user_{uuid.uuid4()}"
    
    # Process query through router
    response = await router.route_query(request.query, session_id)
    
    return {
        "answer": response.get("answer", "No response generated"),
        "session_id": session_id,
        "expert": getattr(response, "expert", "unknown"),
        "sources": response.get("sources", 0)
    }

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