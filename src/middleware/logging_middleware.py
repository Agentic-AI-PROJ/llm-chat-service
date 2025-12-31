import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from src.utils.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests.
    Logs method, URL, status code, and duration in milliseconds.
    Matches the format of Node.js loggingMiddleware.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        method = request.method
        url = request.url.path
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration = int((time.time() - start_time) * 1000)
        status_code = response.status_code
        
        # Log the request using INFO level
        logger.info(f"{method} {url} {status_code} {duration}ms")
        
        return response
