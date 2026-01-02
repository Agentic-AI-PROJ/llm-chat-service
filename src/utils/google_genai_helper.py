
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
from src.utils.logger import logger

def _convert_messages_to_google_format(messages: List[Dict[str, Any]]) -> str:
    """
    Convert OpenAI-style messages to a single string prompt or a list of contents.
    Google GenAI SDK `generate_content` accepts string or list of `types.Content`.
    For simplicity and best grounding performance, we'll concatenate the conversation 
    into a structured string or pass the last user message with context.
    
    However, preserving chat history is better.
    Let's reconstruct the chat history.
    """
    # Simple concatenation for now, as `contents` can be a string.
    # Improve this to use proper Content objects if needed for multi-turn.
    # Given the user example used "contents='Who won...'", a string is fine.
    
    full_prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        full_prompt += f"{role.upper()}: {content}\n"
    
    return full_prompt.strip()

async def generate_with_google(
    model_id: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    enable_grounding: bool = False
) -> Dict[str, Any]:
    """
    Generate content using Google GenAI Native SDK.
    
    Args:
        model_id: e.g. "gemini-2.5-flash" (stripped of 'gemini/' prefix if present)
        api_key: Google API Key
        messages: List of OpenAI format messages
        
    Returns:
        Dict with keys "content", "usage" (if available)
    """
    # Clean model ID
    if "/" in model_id:
        model_id = model_id.split("/")[-1]
        
    client = genai.Client(api_key=api_key)
    
    config_args = {}
    if enable_grounding:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        config_args["tools"] = [grounding_tool]
        
    config = types.GenerateContentConfig(**config_args)
    
    # Convert messages
    # For now, we take the last message as the query for grounding, 
    # but ideally we pass history.
    # The SDK supports list of contents.
    # Let's try passing the full string representation which works well for simple queries.
    prompt = _convert_messages_to_google_format(messages)
    
    # We need to run this synchronously? The client might be sync. 
    # genai.Client() has async methods?
    # User example: client.models.generate_content(...) -> sync.
    # We should wrap it in asyncio.to_thread if it's blocking.
    
    import asyncio
    
    def _call_google():
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config
        )
        
    response = await asyncio.to_thread(_call_google)
    
    return {
        "content": response.text,
        "usage": response.usage_metadata # Might need conversion
    }
