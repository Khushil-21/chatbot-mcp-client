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
        "transport": "streamable_http",
        "url": "https://khushil-expense-tracker.fastmcp.app/mcp",
    },
}


def print_user(msg: str):
    print(f"\n\n👤 You:  {msg}")


def print_assistant(msg: str):
    print(f"\n🤖 Assistant:  {msg}")


def print_tool_call(name, args, tool_id):
    print(f"\n🛠  Tool Call -> {name}  ID -> {tool_id} Args -> {args}")

def print_tool_output(output):
    print("\n📤 Tool Output:",output)


async def main():
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    named_tools = {tool.name: tool for tool in tools}
    print("\nAvailable Tools:")
    for t in named_tools:
        print(f" - {t}")

    llm = ChatOllama(model="gpt-oss:20b-cloud")
    llm_with_tools = llm.bind_tools(tools)

    print("\n💬 MCP Terminal Chat")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("\n\n👤 You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\n👋 Goodbye!\n")
            break

        # print_user(user_input)

        response = await llm_with_tools.ainvoke(user_input)

        # No tool call
        if not response.tool_calls:
            print_assistant(response.content)
            continue

        # Tool calls
        tool_messages = []
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args") or {}
            tool_id = tc["id"]

            print_tool_call(tool_name, tool_args, tool_id)

            result = await named_tools[tool_name].ainvoke(tool_args)
            print_tool_output(result)

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_id,
                )
            )

        # Final assistant message
        final_response = await llm_with_tools.ainvoke(
            [user_input, response, *tool_messages]
        )
        print_assistant(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
