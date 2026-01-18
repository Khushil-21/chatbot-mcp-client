import asyncio
import json
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# ───────────────────────────────────
# MCP Servers Config
# ───────────────────────────────────
SERVERS = {
    "demo-mcp-server": {
        "transport": "stdio",
        "command": "uv",
        "args": [
            "run", "--with", "fastmcp", "fastmcp", "run",
            r"E:\\MyUnquieProjects\\demo-mcp-server\\main.py",
        ],
    },
    "Expense Tracker": {
        "transport": "streamable_http",
        "url": "https://khushil-expense-tracker.fastmcp.app/mcp",
    },
}

SYSTEM_PROMPT = (
    "You have access to tools. When you choose to call a tool, do not narrate status updates. "
    "After tools run, return only a concise final answer."
)

st.set_page_config(page_title="Ollama MCP Chat", layout="centered")
st.title("🧰 Ollama MCP Chat")


# Initialize persistent asyncio event loop
def get_event_loop():
    if "loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.loop = loop
    return st.session_state.loop


loop = get_event_loop()

# One-time init
if "initialized" not in st.session_state:
    # 1) Ollama LLM
    st.session_state.llm = ChatOllama(model="gpt-oss:20b-cloud")

    # 2) MCP tools
    client = MultiServerMCPClient(SERVERS)
    tools = loop.run_until_complete(client.get_tools())
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    # 3) Bind tools
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    # 4) History
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True

# Render history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        if getattr(msg, "tool_calls", None):
            # skip intermediate assistant messages
            continue
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input
user_input = st.chat_input("Type your message…")
if user_input:
    # Append user
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # First pass (tool suggestion)
    first = loop.run_until_complete(
        st.session_state.llm_with_tools.ainvoke(st.session_state.history)
    )

    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools → show reply
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)
    else:
        # 1) Add intermediate assistant message
        st.session_state.history.append(first)

        # 2) Call tools
        tool_msgs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    pass

            tool = st.session_state.tool_by_name[name]
            res = loop.run_until_complete(tool.ainvoke(args))
            tool_msgs.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(res)))

        st.session_state.history.extend(tool_msgs)

        # 3) Final assistant output
        final = loop.run_until_complete(
            st.session_state.llm.ainvoke(st.session_state.history)
        )
        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))
