import logging
import os
from typing import Any, Optional
from colorlog import ColoredFormatter
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for production logs"""
    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = self.formatTime(record, '%Y-%m-%d %H:%M:%S:%f')[:-3]
        log_record['level'] = record.levelname.lower()


def setup_logger() -> logging.Logger:
    """
    Setup logger with winston-style configuration.
    Uses colorized output in development and JSON format in production.
    """
    # Get environment
    environment = os.getenv('ENVIRONMENT', 'development')
    is_development = environment == 'development'
    
    # Create logger
    logger = logging.getLogger('llm-chat-service')
    logger.setLevel(logging.DEBUG if is_development else logging.WARNING)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    if is_development:
        # Colorized format for development
        formatter = ColoredFormatter(
            '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S:%f',
            reset=True,
            log_colors={
                'DEBUG': 'white',
                'INFO': 'green',
                'HTTP': 'magenta',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        # Trim milliseconds to 3 digits
        class CustomColoredFormatter(ColoredFormatter):
            def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
                result = super().formatTime(record, datefmt)
                return result[:-3] + 'ms'
        
        formatter = CustomColoredFormatter(
            '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S:%f',
            reset=True,
            log_colors={
                'DEBUG': 'white',
                'INFO': 'green',
                'HTTP': 'magenta',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    else:
        # JSON format for production
        formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(message)s')
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Add custom HTTP level (between INFO and WARNING)
HTTP_LEVEL = 25
logging.addLevelName(HTTP_LEVEL, 'HTTP')


def http(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    """Log HTTP requests at HTTP level"""
    if self.isEnabledFor(HTTP_LEVEL):
        self.log(HTTP_LEVEL, message, *args, **kwargs)


# Add http method to Logger class
logging.Logger.http = http  # type: ignore

# Create and export logger instance
logger = setup_logger()


