# Setup Instructions 
**For this example we will consider making an agent from search.py**

For windows specific instructions please look [here](https://modelcontextprotocol.io/quickstart/server)

1. Install MCP locally
   ```
   cd personal_mcp_agents
   url -LsSf https://astral.sh/uv/install.sh | sh
   # Create virtual environment and activate it
   uv venv
   source .venv/bin/activate
    
   # Install dependencies
   uv add "mcp[cli]" httpx requests
   uv init search
   mv search.py search/
   ```
4. Edit - ``~/Library/Application\ Support/Claude/claude_desktop_config.json``. 
5. Add the relevant MCP agent (for this example we will integrate the perplexity search agent with claude). Change to your username and put ABSOLUTE PATHS here -
```
{
    "mcpServers": {
        "search": {
            "command": "/Users/username/.local/bin/uv",
            "args": [
                "--directory",
                "path/to/personal_mcp_agents/search",
                "run",
                "search.py"
            ]
        }
    }
}
```
4. Open the claude desktop application.
