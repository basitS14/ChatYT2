from src.helpers import (
    extract_id,
    get_transcript,
    split_transcript,
    QueryResolver,
    TranscriptIndexerQdrant
)

class InsertionPipeline:
    def __init__(self , video_id):
        self.video_id = video_id
        self.resolver = QueryResolver()
        self.indexer = TranscriptIndexerQdrant()
    
    def invoke(self):
        transcript = get_transcript(self.video_id)
        chunks = split_transcript(transcript)
        embeddings, batch_context = self.indexer.generate_vector_embeddings(chunks)

        self.indexer.insert_data(self.video_id , embeddings, batch_context)

        print("Data inserted !!")


class RetrievalPipeline:
    def __init__(self):
        self.resolver = QueryResolver()
        self.indexer = TranscriptIndexerQdrant()
    
    def invoke(self , query):
        retrieved_documents = self.indexer.retrieve_documents(query)
        
        res = self.resolver.get_responce(query , retrieved_documents)
        return res


        