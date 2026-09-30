

"""
Shared components for LangGraph Chatbot frontend applications.
This module contains common functions and components used by both frontend.py and thread_frontend.py
to reduce code duplication and provide consistent behavior.
"""

import streamlit as st
from langchain_core.messages import HumanMessage

# ==========================================================
# Helper Functions
# ==========================================================

def _extract_content(content):
    """
    Extract text from message content, handling different content types.
    Content can be a string, a list of content blocks, dicts, or message objects.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # Handle nested message object containing content attribute
    if hasattr(content, "content") and not isinstance(content, str):
        return _extract_content(content.content)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(_extract_content(block))
        return "".join(text_parts)

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        if "text" in content:
            return str(content["text"])

    # Fallback for other types
    return str(content)


def load_thread_history(thread_id, graph):
    """
    Load message history for a specific thread from LangGraph state.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state = graph.get_state(config)
    except Exception:
        return []

    # Safely handle state.values being None or missing
    messages = []
    if hasattr(state, 'values') and state.values:
        messages = state.values.get("messages", [])

    history = []

    for message in messages:
        msg_type = None
        if hasattr(message, "type"):
            msg_type = message.type
        elif isinstance(message, dict):
            msg_type = message.get("type") or message.get("role")

        msg_content = getattr(message, "content", message.get("content") if isinstance(message, dict) else "")

        if msg_type in ("human", "user"):
            history.append({
                "role": "user",
                "content": _extract_content(msg_content)
            })
        elif msg_type in ("ai", "assistant"):
            history.append({
                "role": "assistant",
                "content": _extract_content(msg_content)
            })

    return history


def display_message_history(message_history):
    """
    Display the chat message history using Streamlit chat components.
    Handles user, assistant, and tool activity messages.
    """
    for message in message_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        if role == "tool":
            with st.expander(f"🛠️ Tool Call: {message.get('name', 'Tool')}", expanded=False):
                st.code(content)
        else:
            with st.chat_message(role):
                st.write(content)


def initialize_session_state():
    """
    Initialize common session state variables.
    Returns: tuple of (message_history, thread_id)
    """
    if "threads" not in st.session_state:
        st.session_state.threads = ["chat_1"]

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "chat_1"

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    return st.session_state.message_history, st.session_state.thread_id


import os


def create_thread_config(thread_id, tags=None, metadata=None):
    """
    Create configuration dictionary for LangGraph with thread_id, tracing tags, and metadata.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "tags": tags or ["streamlit", "langgraph-chatbot"],
        "metadata": metadata or {"thread_id": thread_id, "application": "LangGraph_Chatbot"}
    }
    return config


def render_thread_sidebar(graph, available_tools=None):
    """
    Render thread management sidebar and return current active thread_id.
    """
    st.sidebar.title("💬 My Conversations")

    # New Chat
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_thread = f"chat_{len(st.session_state.threads) + 1}"
        st.session_state.threads.append(new_thread)
        st.session_state.thread_id = new_thread
        st.session_state.message_history = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Threads")

    # Existing thread selector
    for thread in st.session_state.threads:
        is_active = (thread == st.session_state.thread_id)
        label = f"▶ {thread}" if is_active else f"💬 {thread}"
        if st.sidebar.button(label, key=f"btn_{thread}", use_container_width=True):
            st.session_state.thread_id = thread
            st.session_state.message_history = load_thread_history(thread, graph)
            st.rerun()

    st.sidebar.markdown("---")

    # Active Tools Panel
    if available_tools:
        with st.sidebar.expander("🛠️ Active Tools", expanded=False):
            for t in available_tools:
                tool_name = getattr(t, "name", str(t))
                tool_desc = getattr(t, "description", "")
                st.markdown(f"**`{tool_name}`**")
                if tool_desc:
                    st.caption(tool_desc)

    # Display Observability Sidebar
    display_observability_sidebar(
        thread_id=st.session_state.thread_id,
        message_history=st.session_state.message_history,
        last_latency=st.session_state.get("last_latency")
    )

    return st.session_state.thread_id


def display_observability_sidebar(thread_id=None, message_history=None, last_latency=None):
    """
    Render Observability sidebar widget showing tracing status, project, and latency metrics.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "langgraph-chatbot")

    with st.sidebar.expander("📊 Observability & Metrics", expanded=True):
        if tracing_enabled:
            st.markdown(f"**LangSmith Status:** 🟢 `Active`")
            st.markdown(f"**Project:** `{project_name}`")
        else:
            st.markdown(f"**LangSmith Status:** ⚪ `Inactive`")

        if thread_id:
            st.markdown(f"**Active Thread:** `{thread_id}`")

        if message_history is not None:
            st.markdown(f"**Total Messages:** `{len(message_history)}`")

        if last_latency is not None:
            st.markdown(f"**Last Latency:** `{last_latency:.2f}s`")


def handle_api_error(error, context=""):
    """
    Handle API errors in a consistent way across frontend applications.
    """
    error_message = str(error)
    if context:
        st.error(f"Error in {context}: {error_message}")
    else:
        st.error(f"Error: {error_message}")