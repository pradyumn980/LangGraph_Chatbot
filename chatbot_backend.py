import os
import logging
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from huggingface_hub import login

from langchain_core.messages import BaseMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# Logging configuration
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    """
    Process incoming messages and generate AI response.
    """
    try:
        messages = state["messages"]
        logger.info(f"Processing {len(messages)} message(s)")

        response = model.invoke(messages)

        return {
            "messages": [response]
        }

    except ValueError as e:
        # Handle validation errors (e.g., invalid input)
        logger.error(f"Validation error: {e}")
        return {
            "messages": [
                AIMessage(
                    content=f"I couldn't process your request due to invalid input. Please try again."
                )
            ]
        }
    except ConnectionError as e:
        # Handle network/API connectivity issues
        logger.error(f"Connection error: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I'm having trouble connecting to the AI service. Please check your internet connection and try again."
                )
            ]
        }
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error in computation: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I encountered an unexpected error. Please try again later."
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