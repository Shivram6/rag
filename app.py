from fastapi import FastAPI
from pydantic import BaseModel,Field
import os
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from cachetools import TTLCache
chat_cache = TTLCache(maxsize=100, ttl=3600)
load_dotenv()
api_key = os.getenv("api")
pc = Pinecone(api_key=api_key)
index = pc.Index("rag-index")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI(title="jp solutions bot")

class chatreq(BaseModel):
    question:str = Field(..., min_length=2, max_length=500, example="What is the leave policy?")
      
def retrieval(query,top_k=1):
    query_emb = embedder.encode(query,normalize_embeddings=True).tolist()
    result = index.query(vector=query_emb,top_k = top_k, include_metadata=True)
    contexts = [match["metadata"]["text"]for match in result["matches"]]
    return "\n".join(contexts)

@app.get("/")
def home():
    return {"message":"Welcome to JP Solutions Bot API. Use the /char endpoint to ask questions."}


@app.post("/chat")
def chat(Query: chatreq):
    query = Query.question
    if query in chat_cache:
        print(f"DEBUG: Serving from cache: {query}")
        return {"answer": chat_cache[query], "source": "cache"}
    context = retrieval(query)
    system_rules = (
            "You are an internal JP Solutions assistant. "
            "Provide a concise 2-sentence answer based ONLY on the context. "
            "Do not repeat the context or use conversational filler."
        )
        
    user_input = f"Context: {context}\n\nQuestion: {query}"

    response = ollama.chat(
            model="gemma:latest",
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_input}
            ],
            options={"temperature": 0.2} 
        )

    print("\nAnswer:")
    print(response["message"]["content"].strip()) 
    chat_cache[query] = response["message"]["content"].strip()
    return {"answer": response["message"]["content"].strip()}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
