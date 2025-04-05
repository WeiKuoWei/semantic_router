import os
import asyncio
import time
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI

from utils.config import OPENAI_API_KEY, DEFAULT_MODEL

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def generate_response(query: str, context: List[str], expert_name: str = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Generate a response using OpenAI's GPT model with retrieved context.
    
    Args:
        query: The user's query
        context: List of text chunks retrieved from the vector database
        expert_name: Name of the expert (optional)
        model: OpenAI model to use
        
    Returns:
        Dictionary containing the response
    """
    if not OPENAI_API_KEY:
        return {"answer": "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."}
    
    try:
        # Combine context into a single string
        # collection.query returns a list of lists
        if context and isinstance(context[0], list):
            context_text = "\n\n".join([text for sublist in context for text in sublist])
        else:
            context_text = "\n\n".join(context)
        
        
        # Create system prompt based on expert
        if expert_name:
            system_prompt = f"You are an expert in {expert_name}. Answer the question based on the following context."
        else:
            system_prompt = "Answer the question based on the following context."
        
        # Create messages for OpenAI API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
        
        # Call OpenAI API
        start_time = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        logging.info(f"OpenAI API call took {time.time() - start_time:.2f} seconds")
        
        # Extract answer
        answer = response.choices[0].message.content
        
        return {"answer": answer, "sources": len(context[0])}
    
    except Exception as e:
        return {"answer": f"Error generating response: {str(e)}"}