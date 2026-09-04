import streamlit as st
from chatbot_backend import graph
from langchain_core.messages import HumanMessage
from shared_components import (
    _extract_content,
    load_thread_history,
    display_message_history,
    create_thread_config,
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


# =========================================================
# Session State
# =========================================================

if "threads" not in st.session_state:
    st.session_state.threads = ["chat_1"]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_1"

if "message_history" not in st.session_state:
    st.session_state.message_history = load_thread_history(
        st.session_state.thread_id,
        graph
    )


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("💬 My Conversations")


# New Chat
if st.sidebar.button("➕ New Chat"):
    new_thread = f"chat_{len(st.session_state.threads) + 1}"

    st.session_state.threads.append(new_thread)
    st.session_state.thread_id = new_thread
    st.session_state.message_history = []

    st.rerun()


st.sidebar.header("Conversations")


# Existing threads
for thread in st.session_state.threads:
    if st.sidebar.button(
        thread,
        key=f"button_{thread}"
    ):
        st.session_state.thread_id = thread
        st.session_state.message_history = load_thread_history(thread, graph)
        st.rerun()


# =========================================================
# Display Chat History
# =========================================================

display_message_history(st.session_state.message_history)


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    # Add user message to UI history
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    # Thread configuration
    config = create_thread_config(st.session_state.thread_id)

    # Call LangGraph
    try:
        response = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        # Extract AI message content safely
        ai_message = _extract_content(response["messages"][-1].content)

        # Show AI response
        with st.chat_message("assistant"):
            st.write(ai_message)

        # Save AI message to UI history
        st.session_state.message_history.append({
            "role": "assistant",
            "content": ai_message
        })

    except Exception as e:
        handle_api_error(e, "LangGraph invoke")