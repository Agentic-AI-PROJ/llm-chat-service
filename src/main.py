import os
import json
import time
import base64
from urllib.parse import urlparse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from src.middleware.logging_middleware import LoggingMiddleware
from src.utils.logger import logger
from litellm import completion
import litellm
from src.models.chat_request import ChatRequest
from sse_starlette.sse import EventSourceResponse
from src.config.db import connect_to_mongo, close_mongo_connection, get_database
from src.utils.request_tracker import track_request
from src.utils.model_selector import model_selector
from src.routes.model_routes import router as model_router
from src.routes.log_routes import router as log_router
from src.utils.token_counter import count_tokens, count_messages_tokens
from bson import ObjectId
from src.utils.limit_checker import check_grounding_limit



# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="LLM Chat Service",
    description="Service for handling LLM chat interactions",
    version="1.0.0"
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include model management routes
app.include_router(model_router, prefix="/models", tags=["models"])
app.include_router(log_router, prefix="/logs", tags=["logs"])

# Helper to process messages and handle file:// URLs
def process_messages(messages: list[dict]):
    """
    Recursively process messages to handle file:// URLs.
    Reads local files, converts to base64, and updates the message content.
    """
    processed_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("file://"):
                        try:
                            file_path = urlparse(image_url).path
                            with open(file_path, "rb") as image_file:
                                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                                # Detect mime type based on extension
                                ext = os.path.splitext(file_path)[1].lower()
                                mime_type = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
                                
                                part["image_url"]["url"] = f"data:{mime_type};base64,{encoded_string}"
                                logger.info(f"Converted file URL {image_url} to base64")
                        except Exception as e:
                            logger.error(f"Failed to process file URL {image_url}: {str(e)}")
                            # Keep original URL if processing fails, might fail downstream but better than crashing here
                new_content.append(part)
            message["content"] = new_content
        processed_messages.append(message)
    return processed_messages

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return "RUNNING"

@app.get("/db-health")
async def db_health_check():
    """Database health check endpoint"""
    try:
        db = get_database()
        if db is not None:
            # Test existing connection
            await db.command('ping')
        else:
            # Try to connect
            db = await connect_to_mongo()
            if db is None:
                raise Exception("Failed to connect to database")
        
        return "RUNNING"
    except Exception as e:
        logger.error(f"DB Health Check Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# Helper to check for images in messages
def has_images(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False

# Non Streaming LLM Response
@app.post("/non-stream")
async def non_stream(chat_request: ChatRequest, request: Request):
    start_time = time.time()
    
    # Process local file URLs
    chat_request.messages = process_messages(chat_request.messages)
    
    # Determine required input type
    required_input_type = "image" if has_images(chat_request.messages) else "text"
    
    max_retries = 3
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        attempts += 1
        
        # Auto-select model using round-robin with type capability check
        selected_model = await model_selector.get_next_model(required_input_type, chat_request.model_id)
        if not selected_model:
            # If no models at all on first attempt, fail
            if attempts == 1:
                raise HTTPException(status_code=503, detail=f"No models available for input type: {required_input_type}")
            # If ran out of models during retries, stop
            break
        
        model_id = ObjectId(str(selected_model["_id"]))
        log_data = {
            "ai_model_id": model_id,
            "success": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "response_time_ms": 0,
            "input_text": str(chat_request.messages),
            "output_text": "",
            "grounding_enabled": False
        }
        
        try:
            gemini_api_key = selected_model.get("api_key")
            if not gemini_api_key:
                logger.error(f"Missing API key for model {selected_model.get('name')}")
                log_data["error_message"] = "Model API key not configured"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                continue

            # Check grounding enablement and limits
            tools = None
            if chat_request.enable_grounding:
                is_allowed = await check_grounding_limit(str(model_id))
                if not is_allowed:
                    # Fail fast if limit exceeded? Or fallback to no-grounding?
                    # User said "make sure the limit does not cross the limit", implies stopping it.
                    # Since the user requested "enable grounding", better to fail the request or just disable grounding.
                    # Failing is safer to indicate limit reached.
                    raise HTTPException(status_code=429, detail="Daily grounding limit exceeded for this model.")
                
                log_data["grounding_enabled"] = True
                tools = [{"google_search": {}}]

            # Check for max_input_tokens limit
            model_details = selected_model.get("details", {})
            # Check if using native Google Grounding
            if chat_request.enable_grounding:
                 # Verify it is a google model
                 if "gemini" in selected_model.get("model_id", "").lower() or selected_model.get("base_url", "").find("googleapis") != -1:
                     from src.utils.google_genai_helper import generate_with_google
                     
                     logger.info(f"Using Native Google Client for grounding with model {selected_model.get('name')}")
                     
                     # Check limit
                     is_allowed = await check_grounding_limit(
                        model_id=str(model_id),
                        limit_type="websearch"
                     )
                     
                     if not is_allowed:
                         logger.warning(f"Grounding limit exceeded for model {selected_model.get('name')}")
                         raise HTTPException(status_code=429, detail="Daily grounding limit exceeded for this model.")
                     
                     log_data["grounding_enabled"] = True
                     
                     # Call native helper
                     native_response = await generate_with_google(
                         model_id=selected_model.get("model_id"),
                         api_key=gemini_api_key,
                         messages=chat_request.messages,
                         enable_grounding=True
                     )
                     
                     contact = native_response.get("content", "")
                     # Usage parsing if available (native_response['usage'] might be object)
                     usage = native_response.get("usage")
                     if usage:
                         log_data["input_tokens"] = getattr(usage, 'prompt_token_count', 0)
                         log_data["output_tokens"] = getattr(usage, 'candidates_token_count', 0)
                         log_data["total_tokens"] = getattr(usage, 'total_token_count', 0)
                     
                     log_data["success"] = True
                     log_data["response_time_ms"] = (time.time() - start_time) * 1000
                     log_data["output_text"] = contact
                     await track_request(log_data)
                     
                     logger.info(f"Successfully generated grounded response with model {selected_model.get('name')}")
                     return {"data": contact}
            
            # Standard LiteLLM flow for non-grounding or non-Google models
            max_input_tokens = model_details.get("max_input_tokens")
            
            if max_input_tokens:
                model_id_str = selected_model.get("model_id", "gpt-4")
                input_tokens = count_messages_tokens(chat_request.messages, model_id_str)
                
                if input_tokens > max_input_tokens:
                    logger.warning(f"Input tokens ({input_tokens}) exceeded limit ({max_input_tokens}) for model {selected_model.get('name')}")
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Request too long. Input tokens ({input_tokens}) exceeds model limit of {max_input_tokens}."
                    )

            completion_kwargs = {
                "model": selected_model.get("model_id"),
                "messages": chat_request.messages,
                "api_key": gemini_api_key,
                "base_url": selected_model.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                "drop_params": True,
                "stream": False
            }
            if tools:
                completion_kwargs["tools"] = tools

            response = completion(**completion_kwargs)

            logger.info(f"Response: {response}")
            
            # Extract token usage
            if hasattr(response, 'usage') and response.usage:
                log_data["input_tokens"] = getattr(response.usage, 'prompt_tokens', 0)
                log_data["output_tokens"] = getattr(response.usage, 'completion_tokens', 0)
                log_data["total_tokens"] = getattr(response.usage, 'total_tokens', 0)
            
            log_data["success"] = True
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            
            # Track request asynchronously
            contact = response["choices"][0]["message"]["content"]
            log_data["output_text"] = contact
            await track_request(log_data)
            
            logger.info(f"Successfully generated response with model {selected_model.get('name')}")
            return {"data": contact}

        except litellm.exceptions.RateLimitError as e:
            logger.warning(f"Rate Limit with model {selected_model.get('name')}: {str(e)}")
            last_error = e
            log_data["error_message"] = "Rate limit exceeded"
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            continue # Try next model

        except litellm.exceptions.APIConnectionError as e:
            logger.error(f"API Connection Failed with model {selected_model.get('name')}: {str(e)}")
            last_error = e
            log_data["error_message"] = "API connection error"
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            continue # Try next model

        except litellm.exceptions.Timeout as e:
            logger.error(f"API Timeout with model {selected_model.get('name')}: {str(e)}")
            last_error = e
            log_data["error_message"] = "Request timeout"
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            continue # Try next model

        except litellm.exceptions.AuthenticationError as e:
            # Auth errors are likely config issues, but maybe just this model is bad. 
            # We can try next model or fail. Let's fail for Auth errors as it might be a global config issue.
            logger.error(f"Authentication Error with model {selected_model.get('name')}: {str(e)}")
            log_data["error_message"] = "Authentication error"
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            raise HTTPException(status_code=401, detail="Invalid or expired API key")

        except litellm.exceptions.BadRequestError as e:
            # Bad requests (invalid params) will likely fail on all models
            logger.error(f"Bad Request: {str(e)}")
            log_data["error_message"] = str(e)
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            
            error_detail = f"Bad Request: {str(e)}"
            
            # If we happen to know the limit, add it to the message for helpfulness
            model_details = selected_model.get("details", {})
            max_input_tokens = model_details.get("max_input_tokens")
            if max_input_tokens:
                 error_detail += f" (Max input tokens: {max_input_tokens})"
                 
            raise HTTPException(status_code=400, detail=error_detail)
            
        except Exception as e:
            logger.exception(f"Unhandled Exception with model {selected_model.get('name')}: {str(e)}")
            log_data["error_message"] = f"Internal server error: {str(e)}"
            log_data["response_time_ms"] = (time.time() - start_time) * 1000
            await track_request(log_data)
            raise HTTPException(status_code=500, detail="Internal server error occurred")

    # If we fall through the loop, we failed to get a response
    if last_error:
        raise HTTPException(status_code=503, detail=f"All models failed. Last error: {str(last_error)}")
    
    raise HTTPException(status_code=503, detail="Unable to generate response from any model")

# Streaming LLM Response
@app.post("/stream")
async def stream(chat_request: ChatRequest, request: Request):
    start_time = time.time()
    
    # Process local file URLs
    chat_request.messages = process_messages(chat_request.messages)
    
    # Determine required input type
    required_input_type = "image" if has_images(chat_request.messages) else "text"
    
    # Auto-select model using round-robin - Initial check to fail fast if no models
    selected_model_init = await model_selector.get_next_model(required_input_type, chat_request.model_id)
    if not selected_model_init:
        raise HTTPException(status_code=503, detail=f"No models available for input type: {required_input_type}")
    
    # Not using the initial model immediately, logic is inside generator for retry
    
    model_id = ObjectId(str(selected_model_init["_id"])) # Placeholder for logging
    log_data = {
        "ai_model_id": model_id,
        "success": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "response_time_ms": 0,
        "input_text": str(chat_request.messages),
        "output_text": ""
    }
    async def event_generator():
        max_retries = 3
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            attempts += 1
            
            # Auto-select model using round-robin
            selected_model = await model_selector.get_next_model(required_input_type, chat_request.model_id)
            if not selected_model:
                yield {
                    "event": "error",
                    "data": f'{{"detail": "No models available for input type: {required_input_type}"}}'
                }
                return

            model_id = ObjectId(str(selected_model["_id"]))
            
            log_data = {
                "ai_model_id": model_id,
                "success": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "response_time_ms": 0,
                "input_text": str(chat_request.messages),
                "output_text": ""
            }
            
            accumulated_content = ""  # Track response content for token counting
            
            try:
                gemini_api_key = selected_model.get("api_key")
                if not gemini_api_key:
                    logger.error(f"Missing API key for model {selected_model.get('name')}")
                    log_data["error_message"] = "Model API key not configured"
                    log_data["response_time_ms"] = (time.time() - start_time) * 1000
                    await track_request(log_data)
                    continue # Try next model
                
                # Check for max_input_tokens limit
                model_details = selected_model.get("details", {})
                max_input_tokens = model_details.get("max_input_tokens")
                
                if max_input_tokens:
                    model_id_str = selected_model.get("model_id", "gpt-4")
                    input_tokens = count_messages_tokens(chat_request.messages, model_id_str)
                    
                    if input_tokens > max_input_tokens:
                        logger.warning(f"Input tokens ({input_tokens}) exceeded limit ({max_input_tokens}) for model {selected_model.get('name')}")
                        yield {
                            "event": "error",
                            "data": f'{{"detail": "Request too long. Input tokens ({input_tokens}) exceeds model limit of {max_input_tokens}."}}'
                        }
                        return
                
                # We need to catch errors during the *creation* of the stream or the *first chunk*
                # However, litellm returns a generator. The actual API call might happen when we iterate.
                # So we need to wrap the iteration in try/except too.
                
                response = completion(
                    model=selected_model.get("model_id"),
                    messages=chat_request.messages,
                    api_key=gemini_api_key,
                    base_url=selected_model.get("base_url", "https://generativelanguage.googleapis.com/v1beta"),
                    drop_params=True,
                    stream=True
                )

                # If we got here, connection *might* be established, but we don't know for sure until we iterate.
                # Let's try to iterate. If it fails immediately, we catch and retry.
                # If it fails mid-stream, we can't retry cleanly because we might have already sent data to client.
                # But for RateLimit/ConnectionError, it usually happens at start.
                
                stream_started = False
                
                for chunk in response:
                    stream_started = True
                    
                    # Accumulate token usage from chunks if available
                    if hasattr(chunk, 'usage') and chunk.usage:
                        log_data["input_tokens"] = getattr(chunk.usage, 'prompt_tokens', log_data["input_tokens"])
                        log_data["output_tokens"] = getattr(chunk.usage, 'completion_tokens', log_data["output_tokens"])
                        log_data["total_tokens"] = getattr(chunk.usage, 'total_tokens', log_data["total_tokens"])
                    
                    # Accumulate content for manual token counting
                    if hasattr(chunk, 'choices') and chunk.choices:
                        for choice in chunk.choices:
                            if hasattr(choice, 'delta') and choice.delta:
                                content = getattr(choice.delta, 'content', '')
                                if content:
                                    accumulated_content += content
                                    
                                    # Yield structured JSON data
                                    yield {
                                        "event": "message",
                                        "data": json.dumps({"content": content})
                                    }

                
                # If we finished the loop without error, we are done
                # If no usage data from API, manually count tokens
                if log_data["total_tokens"] == 0:
                    model_id_str = selected_model.get("model_id", "gpt-4")
                    log_data["input_tokens"] = count_messages_tokens(chat_request.messages, model_id_str)
                    log_data["output_tokens"] = count_tokens(accumulated_content, model_id_str)
                    log_data["total_tokens"] = log_data["input_tokens"] + log_data["output_tokens"]
                    logger.debug(f"Manually counted tokens - Input: {log_data['input_tokens']}, Output: {log_data['output_tokens']}")
                
                # Track successful streaming request
                log_data["success"] = True
                log_data["output_text"] = accumulated_content
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                return # SUCCESS, exit generator

            except litellm.exceptions.RateLimitError as e:
                logger.warning(f"Rate Limit with model {selected_model.get('name')}: {str(e)}")
                log_data["error_message"] = "Rate limit exceeded"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                last_error = e
                # If stream already started, we can't retry gracefully (client has received partial data).
                if 'stream_started' in locals() and stream_started:
                     yield {
                        "event": "error",
                        "data": '{"detail": "Rate limit exceeded mid-stream."}'
                    }
                     return
                continue # Try next model

            except litellm.exceptions.APIConnectionError as e:
                logger.error(f"API Connection Failed w/ {selected_model.get('name')}: {str(e)}")
                log_data["error_message"] = "API connection error"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                last_error = e
                if 'stream_started' in locals() and stream_started:
                     yield {
                        "event": "error",
                        "data": '{"detail": "Connection failed mid-stream."}'
                    }
                     return
                continue

            except litellm.exceptions.Timeout as e:
                logger.error(f"API Timeout w/ {selected_model.get('name')}: {str(e)}")
                log_data["error_message"] = "Request timeout"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                last_error = e
                if 'stream_started' in locals() and stream_started:
                     yield {
                        "event": "error",
                        "data": '{"detail": "Timeout mid-stream."}'
                    }
                     return
                continue

            except litellm.exceptions.AuthenticationError as e:
                logger.error(f"Authentication Error: {str(e)}")
                log_data["error_message"] = "Authentication error"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                yield {
                    "event": "error",
                    "data": '{"detail": "Invalid or expired API key"}'
                }
                return

            except Exception as e:
                logger.exception(f"Unhandled Exception: {str(e)}")
                log_data["error_message"] = "Internal server error"
                log_data["response_time_ms"] = (time.time() - start_time) * 1000
                await track_request(log_data)
                yield {
                    "event": "error",
                    "data": '{"detail": "Internal server error occurred"}'
                }
                return

        # If loop finishes with no success
        yield {
            "event": "error",
            "data": f'{{"detail": "All models failed. Last error: {str(last_error)}"}}'
        }

    return EventSourceResponse(event_generator())

# Startup event
@app.on_event("startup")
async def startup_event():
    # Connect to MongoDB
    await connect_to_mongo()
    
    # Initialize model selector
    await model_selector.initialize()
    
    port = os.getenv("PORT", "3005")
    logger.info(f"llm-chat-service server running at http://localhost:{port}")

@app.on_event("shutdown")
async def shutdown_event():
    # Close MongoDB connection
    await close_mongo_connection()

    # Add BadRequest handler for stream
    # Note: Stream errors are yielded, so we don't need a specific catch block inside the loop unless we want to enhance it.
    # The existing general Exception handler catch-all might mask specific BadRequests if we don't catch them explicitly in the stream loop.


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "3005"))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
