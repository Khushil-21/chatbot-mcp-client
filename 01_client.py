import asyncio
import json
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

SERVERS = {
    # local server
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
    # remote server
    "Expense Tracker":{
        "transport":"streamable_http",
        "url":"https://khushil-expense-tracker.fastmcp.app/mcp"
    }
    # local server 
    # "Expense Tracker": {
    #     "transport": "stdio",
    #     "command": "uv",
    #     "args": [
    #         "--directory",
    #         r"E:\\MyUnquieProjects\\expense-tracker-mcp",
    #         "run",
    #         "fastmcp",
    #         "run",
    #         "main.py",
    #     ],
    # },
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
        # "Can you list my all expenses for september (01/09/2025 to 30/9/25) (DD/MM/YYY)month and print it in proper readable format",
        "Can you add an expense for 4th Nov 2025 for pizza order bill was 499",
        "Can you list my all the expenses",
        # "Hi Hello can you tell me what all tools do you have ??",
        # "can you roll 2 dice and also add 10 and 15 and give me result for both",
    ]
    for prompt in prompts:
        print("\n---------------- New Iteration ----------------")
        print("\nUser --> ", prompt)

        response = await llm_with_tools.ainvoke(prompt)
        
        if not response.tool_calls:
            print("\nChatBot --> ", response.content)
            print("\nLLM Failed to identify the tool call. Continuing to next prompt")
            print("--- Tool Calling Failed ---\n")
            # continue
        else:
            tool_messages=[]
            print(f"LLM suggested to call total {len(response.tool_calls)} Tools")
            for tool in response.tool_calls:
                selected_tool = tool["name"]
                selected_tool_args = tool["args"] or {}
                selected_tool_id = tool["id"]
                print(
                    f"\n{selected_tool} was called with args = {selected_tool_args} and tool id = {selected_tool_id}"
                )

                tool_result = await named_tools[selected_tool].ainvoke(selected_tool_args)
                tool_messages.append(ToolMessage(content=json.dumps(tool_result,indent=4), tool_call_id=selected_tool_id))
                print("\nTool Result --> ", tool_result)
                

            final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
            print("\nChatBot --> ", final_response.content)
        
        print("\n---------------- End of Iteration ----------------\n")


if __name__ == "__main__":
    asyncio.run(main())
