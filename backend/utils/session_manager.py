import time
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional

class SessionManager:
    """
    Simple session manager to maintain conversation history for users.
    """
    def __init__(self, tracking_dir: str, max_history: int = 5):

        self.tracking_dir = Path(tracking_dir)
        self.sessions = {}  # Store session data by user ID
        self.log_dir = self.tracking_dir / "conversations"
        self.max_history = max_history
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Load existing sessions from disk
        self._load_sessions_from_disk()

    def _load_sessions_from_disk(self):
        """Load existing sessions from disk."""
        # Look for all JSONL files in the log directory
        jsonl_files = glob.glob(str(self.log_dir / "*.jsonl"))
        
        for file_path in jsonl_files:
            # Extract user_id from filename
            filename = os.path.basename(file_path)

            # in user_1743858391229_2025-04-06, get user_1743858391229
            user_id = filename.split("_",2)[0] + "_" + filename.split("_",2)[1]
            
            # Initialize empty session if needed
            if user_id not in self.sessions:
                self.sessions[user_id] = []
            
            # Read messages from file
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            message = json.loads(line.strip())
                            # Only add message entries (not system events)
                            if 'query' in message and 'response' in message:
                                self.sessions[user_id].append(message)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error loading session from {file_path}: {e}")
        
        # Sort messages by timestamp and limit to max_history
        for user_id in self.sessions:
            self.sessions[user_id] = sorted(
                self.sessions[user_id], 
                key=lambda x: x.get('timestamp', 0)
            )[-self.max_history:]

    def add_message(self, user_id: str, query: str, response: Dict[str, Any], conversation_context: str, documents: List[str], expert_name: Optional[str] = None):
        """
        Add a message to the user's session history.
        
        Args:
            user_id: Identifier for the user
            query: The user's query
            response: The system's response
            expert_name: Name of the expert that handled the query (if known)
        """
        timestamp = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        # Initialize session if it doesn't exist
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        # Create message record
        message = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "query": query,
            "response": response.get("answer", "No answer provided"),
            "conversation_context":conversation_context,
            # "documents": documents,
            "expert": expert_name
        }
        
        # Add to session history
        self.sessions[user_id].append(message)
        
        # Log the conversation
        self._log_conversation(user_id, message)
        
        # Keep only the most recent messages (sliding window)
        if len(self.sessions[user_id]) > self.max_history:
            self.sessions[user_id] = self.sessions[user_id][-self.max_history:]
    
    def get_history(self, user_id: str) -> List[Dict]:
        """
        Get the conversation history for a user.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            List of message records for the user
        """
        return self.sessions.get(user_id, [])
    
    def get_formatted_history(self, user_id: str) -> str:
        """
        Get a formatted string of conversation history for LLM context.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            Formatted conversation history string
        """
        history = self.get_history(user_id)
        if not history:
            return ""
            
        formatted_exchanges = []
        for msg in history:
            formatted_exchanges.append(f"User: {msg['query']}")
            formatted_exchanges.append(f"Assistant: {msg['response']}")
            
        return "\n".join(formatted_exchanges)
    
    def get_conversation_context(self, user_id: str) -> str:
        """
        Get a condensed version of conversation history for query context.
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            Condensed conversation context
        """
        history = self.get_history(user_id)
        if not history:
            return ""
            
        # Use just the last few exchanges to keep the context relevant
        context_parts = []
        for msg in history[-min(3, len(history)):]:  # Last 3 exchanges max
            context_parts.append(msg['query'])
            
        return " ".join(context_parts)
    
    def clear_session(self, user_id: str):
        """
        Clear the session history for a user.
        
        Args:
            user_id: Identifier for the user
        """
        if user_id in self.sessions:
            # Log that the session was cleared
            self._log_conversation(user_id, {
                "timestamp": time.time(),
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_event": "Session cleared"
            })
            
            # Clear the session
            self.sessions[user_id] = []
    
    def _log_conversation(self, user_id: str, message: Dict):
        """
        Log a conversation entry to a file.
        
        Args:
            user_id: Identifier for the user
            message: The message record to log
        """
        # Create log file name based on user ID and date
        date_str = time.strftime("%Y-%m-%d", time.localtime())
        log_file = self.log_dir / f"{user_id}_{date_str}.jsonl"
        
        # Append the message to the log file
        with open(log_file, "a") as f:
            f.write(json.dumps(message) + "\n")