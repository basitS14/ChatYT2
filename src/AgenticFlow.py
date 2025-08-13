import os
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex, Document, Settings
from src.helpers import TranscriptIndexerQdrant
from llama_index.core.selectors import PydanticSingleSelector
# from llama_index.core.selectors.pydantic_selectors import Pydantic
from dotenv import load_dotenv

load_dotenv()

class AgenticRAG:
    def __init__(self, video_id: str):
        self.llm = Groq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1000,
            temperature=0.5
        )
        self.video_id = video_id
        self.indexer = TranscriptIndexerQdrant()
        print("idhar")
        # Ensure transcript is indexed
        if not self.indexer.retrieval_by_video_id(video_id):
            self.indexer.insert_transcript(video_id)
        print("abhi idhar")
        # Set global settings
        Settings.embed_model = self.indexer.embed_model
        Settings.llm = self.llm

        # Build index once
        documents = self.indexer.retrieval_by_video_id(self.video_id)
        llama_docs = [
            Document(text=record.payload.get("context", ""), metadata=record.payload)
            for record in documents
        ]
        print("hakla")
        self.vectorindex = VectorStoreIndex.from_documents(llama_docs)
        print("vector_index setup ho gaya")
        self.vector_query_engine = self.vectorindex.as_query_engine()
        print("shahrukh")
        def llm_engine(query):
            return self.llm.complete(query)

        self.list_tool = QueryEngineTool.from_defaults(
        query_engine=llm_engine,
        description="Useful for normal queries like greeting messages.",
        )
        self.vector_tool = QueryEngineTool.from_defaults(
        query_engine=self.vector_query_engine,
        description="Useful for retrieving specific context related to the data source",
        )

        # initialize router query engine (single selection, pydantic)
        self.query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(),
        query_engine_tools=[
            self.list_tool,
            self.vector_tool,
        ],
        )

    def chat(self , query):
        print("hakla shahrukh")
        return self.query_engine.query(query)
        
