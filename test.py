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
from src.prompts import PROMPT_WITH_CONTEXT , PROMPT_FOR_DIRECT
from langchain.schema import SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store 

load_dotenv()

checkpointer = MemorySaver()

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
    distance = models.Distance.DOT,
    content_payload_key="context"
    
)


retriever = store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_transcript_chunks",
    "Search and return relevant information from the transcript",
)

response_model = init_chat_model("groq:openai/gpt-oss-20b" )

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = PROMPT_FOR_DIRECT.format(query=state["messages"][0].content) 

    messages_with_system = [
        SystemMessage(content=prompt),
        *state["messages"]
    ]

    response = (
        response_model
        .bind_tools([retriever_tool]).invoke( messages_with_system)
    )
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = PROMPT_WITH_CONTEXT.format(query=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# memory
# def store_in_memory(state: MessagesState):
#     for msg in state["messages"]:
#         if isinstance(msg, dict):
#             role = msg["role"]
#             content = msg["content"]
#         else:
#             role = msg.type
#             content = msg.content
#         memory_store.put(
#             ("chat_memory",),
#             "namespace",
#             {"role": role, "content": content}
#         )
#     return state

# def recall_from_memory(state: MessagesState):
    query_text = state["messages"][-1].content
    results = memory_store.search(
        ("chat_memory",),
        query=query_text,
        limit=3  # top 3 relevant memories
    )

    retrieved_context = "\n".join(
        [f"{r.value['role']}: {r.value['content']} {r}" for r in results]
    )
    if retrieved_context:
        state["messages"].insert(
            0,
            {"role": "system", "content": f"Relevant past conversation:\n{retrieved_context}"}
        )
    return state


workflow = StateGraph(MessagesState)


# workflow.add_node("recall_from_memory", recall_from_memory)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("generate_answer" , generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools":"retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", END)

graph = workflow.compile(checkpointer=checkpointer)

print("Chat with AI (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Run the workflow with the user message
    retrieved = False
    final_response = ""

    for chunk in graph.stream({"messages": [{"role": "user", "content": user_input}]} ,
                               {"configurable": {"thread_id": "1"}}):
        for node, update in chunk.items():
            # Check if retrieval node was part of the run
            if node == "retrieve":
                retrieved = True

            # Capture the assistant's last message
            if "messages" in update:
                # Get the last message's content
                msg_content = update["messages"][-1].content
                final_response = msg_content

    # Display AI output
    if retrieved:
        print(f"AI [context]: {final_response}\n")
    else:
        print(f"AI: {final_response}\n")
