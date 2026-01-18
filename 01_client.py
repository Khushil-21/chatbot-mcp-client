import asyncio
import json
from langchain_core.messages import ToolMessage
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
    print(named_tools.keys())

    llm = ChatOllama(model="gpt-oss:20b-cloud")
    llm_with_tools = llm.bind_tools(tools)
    prompts = [
        # "roll dice 7 times and do average of that",
        # "add 10 and 995",
        "Can you list my all expenses for september (1/9/25 to 30/9/25) month and print it in proper readable format",
        "Can you list my all expenses for January 2026 month and print it in proper readable format",
    ]
    for prompt in prompts:
        print("\n---------------- New Iteration ----------------")
        print("\nUser --> ", prompt)

        response = await llm_with_tools.ainvoke(prompt)

        selected_tool = response.tool_calls[0]["name"]
        selected_tool_args = response.tool_calls[0]["args"]
        selected_tool_id = response.tool_calls[0]["id"]
        print(
            f"\n{selected_tool} was called with args = {selected_tool_args} and tool id = {selected_tool_id}"
        )

        tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_message = ToolMessage(content=tool_result, tool_call_id=selected_tool_id)
        print("\nTool Result --> ", tool_result)

        final_response = await llm_with_tools.ainvoke([prompt, response, tool_message])
        print("\nChatBot --> ", final_response.content)

        print("\n---------------- End of Iteration ----------------\n")


if __name__ == "__main__":
    asyncio.run(main())
