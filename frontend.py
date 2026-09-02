import streamlit as st
from chatbot_backend import graph
from langchain_core.messages import HumanMessage

# Page configuration
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖"
)

st.title("🤖 LangGraph Chatbot")


# Initialize session state
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-user-1"


# Display previous messages
for message in st.session_state.message_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
user_input = st.chat_input("Ask me anything...")


if user_input:

    # Display user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    # Call LangGraph
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    try:

        response = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config
        )

        ai_message = response["messages"][-1].content

        # Display AI response
        st.session_state.message_history.append({
            "role": "assistant",
            "content": ai_message
        })

        with st.chat_message("assistant"):
            st.write(ai_message)

    except Exception as e:

        st.error(f"Error: {e}")