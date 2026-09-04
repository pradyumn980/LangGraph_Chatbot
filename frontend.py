import streamlit as st
from chatbot_backend import graph
from shared_components import (
    _extract_content,
    load_thread_history,
    display_message_history,
    initialize_session_state,
    create_thread_config,
    handle_api_error
)
from langchain_core.messages import HumanMessage

# Page configuration
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


# Initialize session state and get shared variables
message_history, thread_id = initialize_session_state()


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

    # Create config for LangGraph
    config = create_thread_config(thread_id)

    # Stream response token by token
    try:
        with st.chat_message("assistant"):
            # Use an empty container to update progressively
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
                if message_chunk.content:
                    full_response += _extract_content(message_chunk.content)
                    response_placeholder.write(full_response + "▌")

            # Final write without cursor
            response_placeholder.write(full_response)

        # Append AI message to history
        message_history.append({
            "role": "assistant",
            "content": full_response
        })

    except Exception as e:
        handle_api_error(e, "frontend streaming")
        # Optional: remove the last (failed) user message
        # if message_history and message_history[-1]["role"] == "user":
        #     message_history.pop()