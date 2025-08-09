from src.helpers import   TranscriptIndexerQdrant  , extract_id
from src.Pipeline import InsertionPipeline , RetrievalPipeline

link = "https://www.youtube.com/watch?v=tNZnLkRBYA8"

indexer = TranscriptIndexerQdrant()

insertPipe = InsertionPipeline(link=link)
retrievePipe = RetrievalPipeline()

video_id = extract_id(link=link)

if not indexer.isDataExist(video_id):
    insertPipe.invoke()

    
while True:
    user_input = input("You : ")
    if user_input.lower() in ["exit" ,"quit" , "q" ]:
        break
    response = retrievePipe.invoke(user_input)
    print("Bot: " , response)