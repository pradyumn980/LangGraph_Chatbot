import streamlit as st
from chatbot_backend import graph
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


# =========================================================
# Helper: Load messages from LangGraph
# =========================================================

def load_thread_history(thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = graph.get_state(config)

    messages = state.values.get("messages", [])

    history = []

    for message in messages:

        if message.type == "human":

            history.append({
                "role": "user",
                "content": message.content
            })

        elif message.type == "ai":

            history.append({
                "role": "assistant",
                "content": message.content
            })

    return history


# =========================================================
# Session State
# =========================================================

if "threads" not in st.session_state:
    st.session_state.threads = ["chat_1"]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_1"

if "message_history" not in st.session_state:

    st.session_state.message_history = load_thread_history(
        st.session_state.thread_id
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

    # New thread has no messages
    st.session_state.message_history = []

    st.rerun()


st.sidebar.header("Conversations")


# Existing threads
for thread in st.session_state.threads:

    if st.sidebar.button(
        thread,
        key=f"button_{thread}"
    ):

        # Change current thread
        st.session_state.thread_id = thread

        # Load messages belonging to this thread
        st.session_state.message_history = load_thread_history(
            thread
        )

        st.rerun()


# =========================================================
# Display Chat History
# =========================================================

for message in st.session_state.message_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# =========================================================
# Chat Input
# =========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(user_input)


    # -----------------------------------------------------
    # Add user message to UI history
    # -----------------------------------------------------

    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })


    # -----------------------------------------------------
    # Thread configuration
    # -----------------------------------------------------

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


    # -----------------------------------------------------
    # Call LangGraph
    # -----------------------------------------------------

    try:

        response = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        ai_message = response["messages"][-1].content


        # -------------------------------------------------
        # Show AI response
        # -------------------------------------------------

        with st.chat_message("assistant"):

            st.write(ai_message)


        # -------------------------------------------------
        # Save AI message to UI history
        # -------------------------------------------------

        st.session_state.message_history.append({
            "role": "assistant",
            "content": ai_message
        })


    except Exception as e:

        st.error(f"Error: {e}")