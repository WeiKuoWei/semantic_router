from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    expert: Optional[str] = None
    sources: Optional[int] = None

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]