# utils/logging_utils.py
import logging
import time
import functools
from datetime import datetime
import os
import json
from typing import Callable, Any

class LoggerSetup:
    @staticmethod
    def setup_logger(log_file: str = None) -> logging.Logger:
        """Set up and configure logger"""
        # Create logger
        logger = logging.getLogger('NLPAnalysis')
        logger.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

        # Create file handler if log_file is specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        return logger

class LoggingDecorators:
    @staticmethod
    def log_function(logger: logging.Logger) -> Callable:
        """Decorator to log function calls with parameters and execution time"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Extract class name if method, otherwise use module name
                if args and hasattr(args[0].__class__, func.__name__):
                    class_name = args[0].__class__.__name__
                    full_name = f"{class_name}.{func.__name__}"
                else:
                    full_name = f"{func.__module__}.{func.__name__}"

                # Log function start with parameters
                params = LoggingDecorators._format_parameters(args[1:], kwargs)
                logger.info(f"Starting {full_name} with parameters: {params}")

                # Execute function and time it
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Log successful completion
                    logger.info(
                        f"Completed {full_name} in {execution_time:.2f} seconds"
                    )
                    return result
                except Exception as e:
                    # Log error if function fails
                    logger.error(
                        f"Error in {full_name}: {str(e)}",
                        exc_info=True
                    )
                    raise
            return wrapper
        return decorator

    @staticmethod
    def log_step(step_name: str, logger: logging.Logger) -> Callable:
        """Decorator to log pipeline steps with detailed information"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                logger.info(f"{'='*20} Starting {step_name} {'='*20}")
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Log step completion with statistics if result is available
                    completion_msg = f"Completed {step_name} in {execution_time:.2f} seconds"
                    if hasattr(result, 'shape'):
                        completion_msg += f" | Output shape: {result.shape}"
                    logger.info(f"{completion_msg}\n{'='*50}")
                    
                    return result
                except Exception as e:
                    logger.error(
                        f"Error in {step_name}: {str(e)}",
                        exc_info=True
                    )
                    raise
            return wrapper
        return decorator

    @staticmethod
    def _format_parameters(args: tuple, kwargs: dict) -> str:
        """Format parameters for logging"""
        params = []
        
        # Format positional arguments
        if args:
            for i, arg in enumerate(args):
                params.append(f"arg{i}={LoggingDecorators._format_value(arg)}")
        
        # Format keyword arguments
        for key, value in kwargs.items():
            params.append(f"{key}={LoggingDecorators._format_value(value)}")
        
        return ", ".join(params)

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for logging, handling different types appropriately"""
        if hasattr(value, 'shape'):  # numpy arrays, pandas DataFrames
            return f"shape={value.shape}"
        elif isinstance(value, (list, tuple)):
            return f"len={len(value)}"
        elif isinstance(value, dict):
            return f"dict(keys={list(value.keys())})"
        elif hasattr(value, '__class__'):  # class instances
            return value.__class__.__name__
        return str(value)