import streamlit as st
from chatbot_backend import graph
from langchain_core.messages import HumanMessage

user_input = st.chat_input("You: ", key="input")


if "message_history" not in st.session_state:
    st.session_state.message_history = []

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if user_input:
    
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    
    response=graph.invoke({'messages': [HumanMessage(content=user_input)]})
    ai_message=response['messages'][-1].content
    
    
    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)