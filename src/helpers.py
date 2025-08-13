import os
import re
from uuid import uuid4
import numpy as np
# from pymilvus import MilvusClient , DataType
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from youtube_transcript_api.proxies import WebshareProxyConfig
from dotenv import load_dotenv
from llama_index.core.text_splitter import SentenceSplitter 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.base.llms.types import ChatMessage , MessageRole
from llama_index.core.schema import TextNode

from more_itertools import chunked
from qdrant_client import QdrantClient
from qdrant_client.models import (
        VectorParams,
        Distance , 
        Filter,
        FieldCondition,
        MatchValue ,
        BinaryQuantizationConfig ,
        BinaryQuantization ,
        PointStruct
)

from src.prompts import CHAT_WITHOUT_QUERY, PROMPT_WITH_CONTEXT, PROMPT_WITH_TOOLS


load_dotenv()

def extract_id(link):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", link)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi(
            # proxy_config=WebshareProxyConfig(
            # proxy_username = os.getenv("PROXY_USERNAME"),
            # proxy_password = os.getenv("PROXY_PASSWORD"),
            # )
        )
        print(video_id)
        res = ytt_api.fetch(video_id)
        print("Working...")
        transcript = " ".join([snippet.text for snippet in res])
        print("Transcript Loaded !!!")
        return transcript

    except TranscriptsDisabled:
        print("Transcripts for this video are disabled by the creator")

def split_transcript(transcript):
    splitter = SentenceSplitter()
    chunks  = splitter.split_text(transcript)
    print("splitted text...")
    return chunks

# def generate_vector_embeddings(chunks):
#     embed_model  = HuggingFaceEmbedding(model_name = "BAAI/bge-large-en-v1.5" , 
#                                         trust_remote_code=True , cache_folder='./hf_cache')
#     # embed_model = HuggingFaceEndpointEmbeddings(
#     #     model = "BAAI/bge-large-en-v1.5"
#     # )
    
#     batch_context = []
#     for context in chunked(chunks , 512):
#         # generate fkoat32 vector
#         batch_context.append(context)
#         embeddings = embed_model.get_text_embedding_batch(context)
#         # convert to binary
#         embed_array = np.array(embeddings)
#         binary_embed = np.where(embed_array > 0 , 1 , 0).astype(np.uint8)

#         # conver to bytes array
#         packed_embeds = np.packbits(binary_embed , axis=1)
#         byte_embeds = [vec.tobytes() for vec in packed_embeds]

#     return byte_embeds , batch_context

# class TranscriptIndexer:
#     def __init__(self, db_path="transcript_quantized.db", collection_name="transcripts_embeddings"):
#         self.collection_name = collection_name
#         self.client = MilvusClient(db_path)
#         self.embed_model = HuggingFaceEmbedding(
#             model_name="BAAI/bge-large-en-v1.5",
#             trust_remote_code=True,
#             cache_folder='./hf_cache'
#         )

#         # Define schema
#         schema = self.client.create_schema(auto_id=True, enable_dynamic_fields=True)
#         schema.add_field(field_name="context", datatype=DataType.VARCHAR)
#         schema.add_field(field_name="binary_vector", datatype=DataType.BINARY_VECTOR)

#         # Define index
#         index_params = self.client.prepare_index_params()
#         index_params.add_index(
#             field_name="binary_vector",
#             index_name="binary_vector_index",
#             index_type="BIN_FLAT",
#             metric_type="HAMMING"
#         )

#         # Create collection if it does not exist
#         if not self.client.has_collection(collection_name):
#             self.client.create_collection(
#                 collection_name=collection_name,
#                 schema=schema,
#                 index_params=index_params
#             )

#     def insert_data(self, byte_embeds, batch_context):
#         """
#         Insert binary vectors and context strings into the collection.
#         """
#         data_to_insert = [
#             {"context": context, "binary_vector": binary_vector}
#             for context, binary_vector in zip(batch_context, byte_embeds)
#         ]

#         self.client.insert(
#             collection_name=self.collection_name,
#             data=data_to_insert
#         )

#     def generate_vector_embeddings(self, chunks, chunk_size=512):

#         batch_context = []
#         byte_embeds = []

#         for context_batch in chunked(chunks, chunk_size):
#             batch_context.extend(context_batch)

#             # Generate float32 vectors
#             embeddings = self.embed_model .get_text_embedding_batch(context_batch)

#             # Convert to binary
#             embed_array = np.array(embeddings)
#             binary_embed = np.where(embed_array > 0, 1, 0).astype(np.uint8)

#             # Convert to byte arrays
#             packed_embeds = np.packbits(binary_embed, axis=1)
#             byte_embeds.extend([vec.tobytes() for vec in packed_embeds])

#         return byte_embeds, batch_context
    
#     def retrieve_documents(self , query):
#         query_embedding = self.embed_model(query)

#         quer_embedd_arr = np.array(query_embedding)
#         binary_query_embedd = np.where(quer_embedd_arr > 0 , 1 , 0)

#         binary_query = np.packbits(binary_query_embedd , axis=1)
#         print(binary_query)

#         search_result = self.client.search(
#             collection_name="transcripts_embeddings",
#             data=[binary_query],
#             limit=5,
#             anns_field="binary_vector",
#             search_params={"metric_type":"HAMMING"},
#             output_fields=["context"]
#         )

#         full_context = []
#         for res in search_result:
#             context = res["payload"]["context"]
#             full_context.append(context)
        
#         return full_context

class TranscriptIndexerQdrant:
    def __init__(self, db_path="transcript_quantized.db", collection_name="transcripts_embeddings"):
        self.collection_name = collection_name
        self.client = QdrantClient(host="localhost", port=6333)
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True, 
            cache_folder='./hf_cache'
        )

        #create new collection
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
            collection_name=self.collection_name,
             vectors_config=VectorParams(
            size=1024,
            distance=Distance.DOT,
            on_disk=True,
        ),
            quantization_config=BinaryQuantization(
            binary=BinaryQuantizationConfig(
                always_ram=True,
        ),
    ),
        )
            print("New collection created...")
            
    def generate_vector_embeddings(self, chunks, chunk_size=512):
        
        embeddings = []
        batch_context = []
        for context_batch in chunked(chunks, chunk_size):
            float_embeddings = self.embed_model.get_text_embedding_batch(context_batch)
            batch_context.extend(context_batch)
            embeddings.extend(float_embeddings)
        print("generated vector embeddings...")
        return embeddings, batch_context


    def insert_data(self, video_id,  embeddings, batch_context):

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={ "video_id":video_id, "context": context}
            )
            for vector, context in zip(embeddings, batch_context)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points = points
        )

    def retrieval_by_video_id(self , video_id):
        
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=1,
            with_payload=True,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        )

        return scroll_result[0]
    
    def retrieve_documents(self , query):
        query_embedding = self.embed_model.get_text_embedding(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            limit=5,
            query_vector=query_embedding,
            with_payload=True   
        )
        print("retrieved documents")
        full_context = [hit.payload["context"] for hit in search_result if "context" in hit.payload]
        print("context retrieved")
        return full_context
    
    def insert_transcript(self , video_id):
        transcript = get_transcript(video_id)
        chunks = split_transcript(transcript)
        embeddings, batch_context = self.generate_vector_embeddings(chunks)

        self.insert_data(video_id , embeddings, batch_context)

        print("Data inserted !!")
        


class QueryResolver():
    def __init__(self , video_id):
        self.llm = Groq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens = 1000,
            temperature = 0.5

        )
        self.indexer = TranscriptIndexerQdrant()
        self.prompt = PROMPT_WITH_TOOLS

        if not self.indexer.retrieval_by_video_id(video_id):
            self.indexer.insert_transcript(video_id)

    def chat_with_function_calling(self , user_query):
        # First LLM call - let it decide whether to retrieve
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_query}
            ],
            tools=[self.indexer.retrieve_documents],
            tool_choice="auto"
        )
        
        # Check if LLM wants to retrieve documents
        if response.tool_calls:
            # Execute retrieval
            search_query = response.tool_calls[0].arguments.query
            context = self.indexer.retrieve_documents(search_query)
            
            # Second LLM call with context
            final_response = self.llm.chat_(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": "", "tool_calls": response.tool_calls},
                    {"role": "tool", "content": f"Retrieved context: {context}", "tool_call_id": response.tool_calls[0].id}
                ]
            )
            return final_response.content
        else:
            # Direct response (greeting/casual)
            return response.content
    
    def get_responce( self, query , full_context):
        prompt = self.prompt_template.format(
            context= full_context,
            query=query
        )
        user_msg = ChatMessage(role=MessageRole.USER , content=prompt)

        response = self.llm.stream_complete(user_msg.content)
        
        full_response = ""
        for chunk in response:
            full_response += chunk.delta # use chunk.text if .delta doesn't exist
        return full_response

        
    

               
            
        
