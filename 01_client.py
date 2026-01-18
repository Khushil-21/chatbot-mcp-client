import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

SERVERS = {
    "demo-mcp-server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--with",
            "fastmcp",
            "fastmcp",
            "run",
            r"E:\\MyUnquieProjects\\demo-mcp-server\\main.py",
        ],
    },
    "Expense Tracker": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "--directory",
            r"E:\\MyUnquieProjects\\expense-tracker-mcp",
            "run",
            "fastmcp",
            "run",
            "main.py",
        ],
    },
}


async def main():

    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    # print(tools)
    named_tools = {tool.name: tool for tool in tools}
    # print(named_tools)
    
    llm=ChatOllama(model="llama3.2:1b")
    llm_with_tools=llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke("list my all expenses for January Month")
    print("Response for expenses ",response)
    print()
    response = await llm_with_tools.ainvoke("roll dice 3 times")
    print("Response for rolling a dice ",response)
    print()
    


if __name__ == "__main__":
    asyncio.run(main())
