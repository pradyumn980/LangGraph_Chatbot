

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
    """
    for message in message_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def initialize_session_state():
    """
    Initialize common session state variables.
    Returns: tuple of (message_history, thread_id)
    """
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"

    return st.session_state.message_history, st.session_state.thread_id


def create_thread_config(thread_id):
    """
    Create configuration dictionary for LangGraph with the specified thread_id.
    """
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def handle_api_error(error, context=""):
    """
    Handle API errors in a consistent way across frontend applications.
    """
    error_message = str(error)
    if context:
        st.error(f"Error in {context}: {error_message}")
    else:
        st.error(f"Error: {error_message}")