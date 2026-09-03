import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from huggingface_hub import login

from langchain_core.messages import BaseMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# Load environment variables
# ==========================================

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file")

login(token=hf_token)


# ==========================================
# HuggingFace Model
# ==========================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)


# ==========================================
# State
# ==========================================

class State(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ==========================================
# Node
# ==========================================

def computation(state: State):

    try:

        messages = state["messages"]

        response = model.invoke(messages)

        return {
            "messages": [response]
        }

    except Exception as e:

        return {
            "messages": [
                AIMessage(
                    content=f"Error: {str(e)}"
                )
            ]
        }


# ==========================================
# Build Graph
# ==========================================

graph_builder = StateGraph(State)

graph_builder.add_node(
    "computation",
    computation
)

graph_builder.add_edge(
    START,
    "computation"
)

graph_builder.add_edge(
    "computation",
    END
)


# ==========================================
# Memory
# ==========================================

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)