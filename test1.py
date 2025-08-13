from operator import index
import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
# from src.helpers import TranscriptIndexerQdrant
from langchain_qdrant import QdrantVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from src.prompts import PROMPT_WITH_CONTEXT, PROMPT_FOR_DIRECT
from langchain.schema import AIMessage


# Define prompts inline

from langchain.schema import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store 

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    cache_folder="./hf_cache"
)
print("embedding set")

client = QdrantClient(host="localhost", port=6333)
print("client set")
store = QdrantVectorStore(
    client=client,
    collection_name="transcripts_embeddings",
    embedding=embeddings,
    distance=models.Distance.DOT,
    content_payload_key="context"
)

memory_store = InMemoryStore(
    index={  # Store extracted memories 
        "dims": 1024,
        "embed": embeddings,
    }
)

manage_memory_tool = create_manage_memory_tool(store=memory_store, namespace="chat_memory")
search_memory_tool = create_search_memory_tool(store=memory_store, namespace='chat_memory')

retriever = store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_transcript_chunks",
    "Search and return relevant information from the transcript",
)

response_model = init_chat_model("groq:openai/gpt-oss-20b")

# Define all tools in one place
tools = [retriever_tool, manage_memory_tool, search_memory_tool]

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    # Get the user's question from the messages
    user_message = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'type') and msg.type == "human":
            user_message = msg
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg
            break
    
    if not user_message:
        # Fallback to the last message
        user_message = state["messages"][-1]
    
    query_content = user_message.content if hasattr(user_message, 'content') else user_message.get("content", "")
    
    # Create a more specific prompt that encourages tool use for memory queries
    enhanced_prompt = f"""
You are an AI assistant with access to tools. Analyze this user query carefully:

"{query_content}"

Available tools:
- retrieve_transcript_chunks: Search transcripts for information
- manage_memory_tool: Store information for later recall
- search_memory_tool: Search for previously stored information

Decision rules:
1. If user asks "what's my name?" or similar personal questions → ALWAYS use search_memory_tool
2. If user shares personal info (name, preferences, etc.) → use manage_memory_tool
3. If question needs transcript knowledge → use retrieve_transcript_chunks
4. Only answer directly if you're certain no tools are needed

What tool should you use for this query?
"""

    messages_with_system = [
        SystemMessage(content=enhanced_prompt)
    ] + state["messages"]

    response = response_model.bind_tools(tools).invoke(messages_with_system)
    return {"messages": [response]}

def generate_answer(state: MessagesState):
    """Generate an answer using the retrieved context."""
    # Get the original user question
    user_message = None
    for msg in state["messages"]:
        if hasattr(msg, 'type') and msg.type == "human":
            user_message = msg
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg
            break
    
    question = user_message.content if user_message else "No question found"
    
    # Get the context from tool calls (last message should contain tool results)
    context = ""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'content'):
        context = last_message.content
    elif isinstance(last_message, dict):
        context = last_message.get("content", "")
    
    # If the context contains memory search results, format it better
    if "search_memory_tool" in str(context) or "manage_memory_tool" in str(context):
        # This is memory-related, create a direct response
        response_content = f"Based on what I remember: {context}"        
        response = AIMessage(content=response_content)
    else:
        # Use the original prompt-based approach for other contexts
        prompt = PROMPT_WITH_CONTEXT.format(query=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}

def store_in_memory(state: MessagesState):
    """Store conversation in memory for future reference."""
    # Only store the most recent human and AI messages to avoid duplicates
    recent_messages = state["messages"][-2:]  # Get last 2 messages
    
    for msg in recent_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
        else:
            role = getattr(msg, 'type', 'unknown')
            content = getattr(msg, 'content', '')
        
        # Map message types consistently
        if role in ["human", "user"]:
            role = "user"
        elif role in ["ai", "assistant"]:
            role = "assistant"
        
        if content and role in ["user", "assistant"]:
            try:
                # Use a timestamp-based key to ensure uniqueness
                import time
                unique_key = f"msg_{int(time.time() * 1000000)}"
                
                memory_store.put(
                    ("chat_memory",),
                    unique_key,
                    {"role": role, "content": content}
                )
                print(f"Stored in memory: {role} - {content[:50]}...")
            except Exception as e:
                print(f"Error storing in memory: {e}")
    return state

# Create the workflow
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("store_in_memory", store_in_memory)

# Add edges
workflow.add_edge(START, "generate_query_or_respond")

# Add conditional edge from generate_query_or_respond
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "tools",  # If tools are called, go to tools node
        END: "store_in_memory",  # If no tools needed, go directly to store_in_memory
    },
)

# From tools node, go to generate_answer to process the retrieved context
workflow.add_edge("tools", "generate_answer")

# After generating answer, store in memory
workflow.add_edge("generate_answer", "store_in_memory")

# After storing in memory, end the workflow
workflow.add_edge("store_in_memory", END)

# Compile the graph
graph = workflow.compile()

print("Chat with AI (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Run the workflow with the user message
    retrieved = False
    final_response = ""

    try:
        for chunk in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for node, update in chunk.items():
                # Check if tools node was part of the run
                if node == "tools":
                    retrieved = True

                # Capture the assistant's last message
                if "messages" in update and update["messages"]:
                    # Get the last message's content
                    last_msg = update["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        msg_content = last_msg.content
                    elif isinstance(last_msg, dict):
                        msg_content = last_msg.get("content", "")
                    else:
                        msg_content = str(last_msg)
                    
                    # Only update final_response if this is from a response-generating node
                    if node in ["generate_query_or_respond", "generate_answer"]:
                        final_response = msg_content

        # Display AI output
        if retrieved:
            print(f"AI [with context]: {final_response}\n")
        else:
            print(f"AI: {final_response}\n")
            
    except Exception as e:
        print(f"Error running workflow: {e}\n")