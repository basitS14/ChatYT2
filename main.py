from src.helpers import   TranscriptIndexerQdrant  , extract_id , QueryResolver
from src.Pipeline import InsertionPipeline , RetrievalPipeline
from src.AgenticFlow import AgenticRAG

link = "https://www.youtube.com/watch?v=tNZnLkRBYA8"


# retrievePipe = RetrievalPipeline()

video_id = extract_id(link=link)
# rag = QueryResolver(video_id=video_id)
rag = AgenticRAG(video_id)
    
while True:
    user_input = input("You : ")
    if user_input.lower() in ["exit" ,"quit" , "q" ]:
        break
    response = rag.chat(user_input)
    print("Bot: " , response)