from typing import Any, Dict, List, Optional
import json
import requests
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("search")

# Constants
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
API_KEY = "ADD_YOUR_API_KEY"

# Available models
SUPPORTED_MODELS = [
    "sonar-deep-research",
    "sonar-reasoning-pro",
    "sonar-reasoning",
    "sonar-pro",
    "sonar",
    "r1-1776"
]

def make_perplexity_request(
    query: str,
    model: str = "sonar",
    system_prompt: str = "Be precise and concise.",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    search_domain_filter: Optional[List[str]] = None,
    return_images: bool = False,
    return_related_questions: bool = False,
    search_recency_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a request to the Perplexity API with proper error handling."""
    
    # Validate model choice
    if model not in SUPPORTED_MODELS:
        return {"error": f"Invalid model. Supported models are: {', '.join(SUPPORTED_MODELS)}"}
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "search_domain_filter": search_domain_filter,
        "return_images": return_images,
        "return_related_questions": return_related_questions,
        "search_recency_filter": search_recency_filter,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def search(query: str, model: str = "sonar") -> str:
    """
    Search the web using Perplexity AI with a specified model.
    
    Args:
        query: The search query
        model: The Perplexity model to use (sonar, sonar-pro, sonar-reasoning, etc.)
    """
    response = make_perplexity_request(query, model=model)
    
    if "error" in response:
        return f"Error: {response['error']}"
    
    if "choices" not in response or not response["choices"]:
        return "No results found."
    
    result = response["choices"][0]["message"]["content"]
    
    # Include citations if available
    if "citations" in response and response["citations"]:
        result += "\n\nSources:\n"
        for i, citation in enumerate(response["citations"], 1):
            result += f"{i}. {citation}\n"
    
    return result

@mcp.tool()
def search_with_domains(query: str, domains: List[str], model: str = "sonar") -> str:
    """
    Search the web using Perplexity AI with domain filtering.
    
    Args:
        query: The search query
        domains: List of domains to filter results (max 3)
        model: The Perplexity model to use (sonar, sonar-pro, sonar-reasoning, etc.)
    """
    if len(domains) > 3:
        return "Error: Maximum of 3 domains allowed for filtering."
    
    response = make_perplexity_request(query, model=model, search_domain_filter=domains)
    
    if "error" in response:
        return f"Error: {response['error']}"
    
    if "choices" not in response or not response["choices"]:
        return "No results found."
    
    result = response["choices"][0]["message"]["content"]
    
    # Include citations if available
    if "citations" in response and response["citations"]:
        result += "\n\nSources:\n"
        for i, citation in enumerate(response["citations"], 1):
            result += f"{i}. {citation}\n"
    
    return result

@mcp.tool()
def search_recent(query: str, time_period: str, model: str = "sonar") -> str:
    """
    Search the web using Perplexity AI with time-based filtering.
    
    Args:
        query: The search query
        time_period: Time filter (month, week, day, hour)
        model: The Perplexity model to use (sonar, sonar-pro, sonar-reasoning, etc.)
    """
    if time_period not in ["month", "week", "day", "hour"]:
        return "Error: Time period must be one of: month, week, day, hour"
    
    response = make_perplexity_request(query, model=model, search_recency_filter=time_period)
    
    if "error" in response:
        return f"Error: {response['error']}"
    
    if "choices" not in response or not response["choices"]:
        return "No results found."
    
    result = response["choices"][0]["message"]["content"]
    
    # Include citations if available
    if "citations" in response and response["citations"]:
        result += "\n\nSources:\n"
        for i, citation in enumerate(response["citations"], 1):
            result += f"{i}. {citation}\n"
    
    return result

@mcp.tool()
def advanced_search(
    query: str, 
    model: str = "sonar",
    system_prompt: str = "Be precise and concise.",
    max_tokens: int = 1000, 
    temperature: float = 0.2,
    top_p: float = 0.9,
    return_related: bool = False,
    return_images: bool = False
) -> str:
    """
    Advanced search with the Perplexity API with custom parameters.
    
    Args:
        query: The search query
        model: The Perplexity model to use (sonar, sonar-pro, sonar-reasoning, etc.)
        system_prompt: Custom system prompt to guide the response
        max_tokens: Maximum tokens in response (100-8000)
        temperature: Randomness of response (0.0-2.0)
        top_p: Nucleus sampling threshold (0.0-1.0)
        return_related: Whether to return related questions
        return_images: Whether to return images (requires appropriate tier)
    """
    response = make_perplexity_request(
        query,
        model=model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        return_related_questions=return_related,
        return_images=return_images
    )
    
    if "error" in response:
        return f"Error: {response['error']}"
    
    if "choices" not in response or not response["choices"]:
        return "No results found."
    
    result = response["choices"][0]["message"]["content"]
    
    # Include citations if available
    if "citations" in response and response["citations"]:
        result += "\n\nSources:\n"
        for i, citation in enumerate(response["citations"], 1):
            result += f"{i}. {citation}\n"
    
    # Include related questions if requested and available
    if return_related and "related_questions" in response and response["related_questions"]:
        result += "\n\nRelated Questions:\n"
        for i, question in enumerate(response["related_questions"], 1):
            result += f"{i}. {question}\n"
    
    return result

@mcp.tool()
def search_with_conversation(
    query: str,
    conversation_history: List[Dict[str, str]],
    model: str = "sonar"
) -> str:
    """
    Search with conversation history for follow-up questions.
    
    Args:
        query: The current search query
        conversation_history: List of previous messages in the format [{"role": "user", "content": "query"}, {"role": "assistant", "content": "response"}]
        model: The Perplexity model to use (sonar, sonar-pro, sonar-reasoning, etc.)
    """
    # Validate model choice
    if model not in SUPPORTED_MODELS:
        return f"Error: Invalid model. Supported models are: {', '.join(SUPPORTED_MODELS)}"
    
    # Prepare the messages array
    messages = [{"role": "system", "content": "Be precise and concise."}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": query})
    
    # Create the payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        if "choices" not in response_data or not response_data["choices"]:
            return "No results found."
        
        result = response_data["choices"][0]["message"]["content"]
        
        # Include citations if available
        if "citations" in response_data and response_data["citations"]:
            result += "\n\nSources:\n"
            for i, citation in enumerate(response_data["citations"], 1):
                result += f"{i}. {citation}\n"
        
        return result
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def get_supported_models() -> List[str]:
    """
    Get a list of supported Perplexity models.
    
    Returns:
        List of supported model names
    """
    return SUPPORTED_MODELS

@mcp.tool()
def get_model_info(model: str) -> str:
    """
    Get information about a specific Perplexity model.
    
    Args:
        model: The model name
    
    Returns:
        Information about the model
    """
    model_info = {
        "sonar-deep-research": {
            "description": "Deep Research conducts comprehensive, expert-level research and synthesizes it into accessible, actionable reports.",
            "features": [
                "Exhaustive Research: Performs dozens of searches, reading hundreds of sources.",
                "Expert-level Analysis: Reasons autonomously and generates detailed insights.",
                "Report Generation: Synthesizes all research into a clear and comprehensive report.",
                "Context Length: 128k",
                "Max Output Tokens: 4k"
            ]
        },
        "sonar-reasoning-pro": {
            "description": "Premier reasoning offering powered by DeepSeek R1 with Chain of Thought (CoT).",
            "features": [
                "Includes detailed Chain of Thought reasoning in responses",
                "Context Length: 128k",
                "Max Output Tokens: 8k"
            ]
        },
        "sonar-reasoning": {
            "description": "Reasoning offering with Chain of Thought (CoT).",
            "features": [
                "Includes detailed Chain of Thought reasoning in responses",
                "Context Length: 128k",
                "Max Output Tokens: 4k"
            ]
        },
        "sonar-pro": {
            "description": "Premier search offering with search grounding, supporting advanced queries and follow-ups.",
            "features": [
                "Context Length: 200k",
                "Max Output Tokens: 8k"
            ]
        },
        "sonar": {
            "description": "Lightweight offering with search grounding, quicker and cheaper than Sonar Pro.",
            "features": [
                "Context Length: 128k",
                "Max Output Tokens: 4k"
            ]
        },
        "r1-1776": {
            "description": "R1-1776 is a version of the DeepSeek R1 model that has been post-trained to provide uncensored, unbiased, and factual information.",
            "features": [
                "Offline chat model (does not use search subsystem)",
                "Context Length: 128k",
                "Max Output Tokens: 4k"
            ]
        }
    }
    
    if model not in model_info:
        return f"Error: Model '{model}' not found. Supported models are: {', '.join(SUPPORTED_MODELS)}"
    
    info = model_info[model]
    result = f"Model: {model}\n\nDescription: {info['description']}\n\nFeatures:\n"
    for feature in info["features"]:
        result += f"- {feature}\n"
    
    return result

if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        import sys
        print(f"Fatal error in MCP server: {e}", file=sys.stderr)

