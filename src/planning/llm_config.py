"""
Configuration file for LLM API settings.
This allows easy switching between different LLM providers.
"""

import os
from typing import Dict, Any

def get_llm_config(provider: str = "lambda_labs") -> Dict[str, Any]:
    """
    Get LLM configuration for the specified provider.
    
    Args:
        provider: LLM provider ("lambda_labs")
        
    Returns:
        Dictionary containing provider configuration
    """
    if provider == "lambda_labs":
        return {
            "api_base": "https://api.lambda.ai/v1",
            "model": "llama-4-maverick-17b-128e-instruct-fp8",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def get_api_key(provider: str = "lambda_labs") -> str:
    """
    Get API key for the specified provider.
    
    Args:
        provider: LLM provider ("lambda_labs")
        
    Returns:
        API key string
    """
    if provider == "lambda_labs":
        api_key = os.getenv("LAMBDA_API_KEY")
        if not api_key:
            raise ValueError("LAMBDA_API_KEY environment variable not set")
        return api_key
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def validate_config(provider: str = "lambda_labs") -> bool:
    """
    Validate that the configuration is properly set up.
    
    Args:
        provider: LLM provider ("lambda_labs")
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        if provider == "lambda_labs":
            api_key = os.getenv("LAMBDA_API_KEY")
            return api_key is not None and len(api_key.strip()) > 0
        else:
            return False
    except Exception:
        return False 