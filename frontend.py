import time
import streamlit as st
from chatbot_backend import graph, tools
from shared_components import (
    _extract_content,
    load_thread_history,
    display_message_history,
    initialize_session_state,
    create_thread_config,
    render_thread_sidebar,
    handle_api_error
)
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


# Initialize session state and get shared variables
message_history, thread_id = initialize_session_state()

# Render Multi-Thread Sidebar & Observability Dashboard with Active Tools list
thread_id = render_thread_sidebar(graph, available_tools=tools)

# Loading thread history from LangGraph checkpointer if message history is empty 
if not message_history:
    loaded_history = load_thread_history(thread_id, graph)
    if loaded_history:
        message_history.extend(loaded_history)
        st.session_state.message_history = message_history

# Display previous messages
display_message_history(message_history)


# Chat input
user_input = st.chat_input("Ask me anything...")


if user_input:
    # Add user message to UI history
    message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Create config for LangGraph with observability metadata
    config = create_thread_config(thread_id, tags=["frontend-stream", "streamlit"])

    start_time = time.perf_counter()

    try:
        with st.chat_message("assistant"):
            status_container = st.empty()
            response_placeholder = st.empty()
            full_response = ""

            for event in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config,
                stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            tool_calls = getattr(msg, "tool_calls", [])
                            if tool_calls:
                                tool_names = ", ".join([tc.get("name", "tool") for tc in tool_calls])
                                status_container.info(f"⚙️ Calling tool(s): `{tool_names}`...")
                            elif msg.content:
                                text = _extract_content(msg.content)
                                if text:
                                    full_response = text
                                    response_placeholder.markdown(full_response)
                        elif isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, "name", "tool")
                            status_container.success(f"✅ Executed tool `{tool_name}`")

            # Clear temporary status and ensure final text is displayed
            status_container.empty()
            if full_response:
                response_placeholder.markdown(full_response)

        elapsed = time.perf_counter() - start_time
        st.session_state.last_latency = elapsed

        # Append AI message to history
        if full_response:
            message_history.append({
                "role": "assistant",
                "content": full_response
            })

    except Exception as e:
        handle_api_error(e, "frontend execution")
        if message_history and message_history[-1]["role"] == "user":
            message_history.pop()
