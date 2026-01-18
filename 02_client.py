import asyncio
import json
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# ─────────────────────────────────────────────
# MCP Servers Config
# ─────────────────────────────────────────────
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
    "After tools run, return only a concise final answer. "
    "Whenever you use dates in tool calls, use the format (dd/mm/yyyy)."
)

st.set_page_config(page_title="Ollama MCP Chat", layout="centered")
st.title("🧰 Ollama MCP Chat")

# ─────────────────────────────────────────────
# Persistent asyncio loop (prevents Event loop closed)
# ─────────────────────────────────────────────
def get_event_loop():
    if "loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.loop = loop
    return st.session_state.loop


loop = get_event_loop()

# ─────────────────────────────────────────────
# One-time initialization
# ─────────────────────────────────────────────
if "initialized" not in st.session_state:
    # LLM
    st.session_state.llm = ChatOllama(model="gpt-oss:20b-cloud")

    # MCP client + tools
    client = MultiServerMCPClient(SERVERS)
    tools = loop.run_until_complete(client.get_tools())
    st.session_state.tools = tools
    st.session_state.tool_by_name = {t.name: t for t in tools}

    # Bind tools to LLM
    st.session_state.llm_with_tools = st.session_state.llm.bind_tools(tools)

    # Conversation history (SystemMessage first)
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.initialized = True


# ─────────────────────────────────────────────
# Render history (User, Tool Calls UI, Assistant)
# ─────────────────────────────────────────────
i = 0
while i < len(st.session_state.history):
    msg = st.session_state.history[i]

    # Skip system messages (not shown in UI)
    if isinstance(msg, SystemMessage):
        i += 1
        continue

    # User message
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
        i += 1
        continue

    # Skip raw ToolMessages here (they are rendered inside AI expander)
    if isinstance(msg, ToolMessage):
        i += 1
        continue

    # Assistant message
    if isinstance(msg, AIMessage):
        tool_calls = getattr(msg, "tool_calls", None)

        # Assistant message that triggered tools
        if tool_calls:
            tool_outputs_for_ui = []
            j = i + 1

            # Collect ToolMessages following this AIMessage
            while j < len(st.session_state.history) and isinstance(
                st.session_state.history[j], ToolMessage
            ):
                tool_msg = st.session_state.history[j]
                try:
                    pretty_output = json.dumps(json.loads(tool_msg.content), indent=2)
                except Exception:
                    pretty_output = tool_msg.content
                tool_outputs_for_ui.append(pretty_output)
                j += 1

            # Render Tool UI
            with st.expander("🔧 Tool Calls & Outputs", expanded=False):
                for tc, pretty_output in zip(tool_calls, tool_outputs_for_ui):
                    tool_name = tc["name"]
                    args = tc.get("args") or {}

                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass

                    try:
                        pretty_args = json.dumps(args, indent=2)
                    except Exception:
                        pretty_args = str(args)

                    st.markdown(f"### 🛠 {tool_name}")
                    st.markdown("**Arguments:**")
                    st.code(pretty_args, language="json")
                    st.markdown("**Output:**")
                    st.code(pretty_output, language="json")

            # Skip AI message and its ToolMessages
            i = j
            continue

        # Normal assistant message (final answer)
        with st.chat_message("assistant"):
            st.markdown(msg.content)
        i += 1
        continue


# ─────────────────────────────────────────────
# Chat input handling
# ─────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input:
    # 1. Append user message
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. First LLM pass – decide tool calls
    first = loop.run_until_complete(
        st.session_state.llm_with_tools.ainvoke(st.session_state.history)
    )
    tool_calls = getattr(first, "tool_calls", None)

    if not tool_calls:
        # No tools used
        with st.chat_message("assistant"):
            st.markdown(first.content or "")
        st.session_state.history.append(first)

    else:
        # Append intermediate assistant message (contains tool_calls, not rendered directly)
        st.session_state.history.append(first)

        tool_outputs_for_ui = []

        # 3. Execute tools
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            tool = st.session_state.tool_by_name[name]
            res = loop.run_until_complete(tool.ainvoke(args))

            # Save tool output to history
            tool_msg = ToolMessage(
                tool_call_id=tc["id"],
                content=json.dumps(res),
            )
            st.session_state.history.append(tool_msg)

            # Pretty formatting for immediate UI
            try:
                pretty_args = json.dumps(args, indent=2)
            except Exception:
                pretty_args = str(args)

            try:
                pretty_output = json.dumps(json.loads(tool_msg.content), indent=2)
            except Exception:
                pretty_output = tool_msg.content

            tool_outputs_for_ui.append((name, pretty_args, pretty_output))

        # 4. Render Tool UI for current turn
        if tool_outputs_for_ui:
            with st.expander("🔧 Tool Calls & Outputs", expanded=True):
                for tool_name, pretty_args, pretty_output in tool_outputs_for_ui:
                    st.markdown(f"### 🛠 {tool_name}")
                    st.markdown("**Arguments:**")
                    st.code(pretty_args, language="json")
                    st.markdown("**Output:**")
                    st.code(pretty_output, language="json")

        # 5. Final assistant answer
        final = loop.run_until_complete(
            st.session_state.llm.ainvoke(st.session_state.history)
        )
        with st.chat_message("assistant"):
            st.markdown(final.content or "")
        st.session_state.history.append(AIMessage(content=final.content or ""))
