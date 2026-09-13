import time
import streamlit as st
from chatbot_backend import graph
from langchain_core.messages import HumanMessage
from shared_components import (
    _extract_content,
    load_thread_history,
    display_message_history,
    initialize_session_state,
    create_thread_config,
    render_thread_sidebar,
    handle_api_error
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


# Initialize session state and get shared variables
message_history, thread_id = initialize_session_state()

# Render Multi-Thread Sidebar & Observability Dashboard
thread_id = render_thread_sidebar(graph)

# Load thread history from LangGraph checkpointer if message history is empty
if not message_history:
    loaded_history = load_thread_history(thread_id, graph)
    if loaded_history:
        message_history.extend(loaded_history)
        st.session_state.message_history = message_history


# =========================================================
# Display Chat History
# =========================================================

display_message_history(st.session_state.message_history)


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:
    # Add user message to UI history
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Thread configuration with observability metadata
    config = create_thread_config(st.session_state.thread_id, tags=["thread-stream", "streamlit"])

    start_time = time.perf_counter()

    # Stream response token by token
    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            for message_chunk, _metadata in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config,
                stream_mode="messages"
            ):
                extracted = _extract_content(getattr(message_chunk, "content", message_chunk))
                if extracted:
                    if extracted.startswith(full_response) and len(extracted) > len(full_response):
                        full_response = extracted
                    elif not full_response.endswith(extracted):
                        full_response += extracted
                    response_placeholder.write(full_response + "▌")

            response_placeholder.write(full_response)

        elapsed = time.perf_counter() - start_time
        st.session_state.last_latency = elapsed

        if full_response:
            st.session_state.message_history.append({
                "role": "assistant",
                "content": full_response
            })

    except Exception as e:
        handle_api_error(e, "LangGraph stream")
        if st.session_state.message_history and st.session_state.message_history[-1]["role"] == "user":
            st.session_state.message_history.pop()