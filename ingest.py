from pinecone import Pinecone
import os 
import re
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
API = os.getenv("api")
pc = Pinecone(api_key=API)
index = pc.Index("rag-index")
with open ("company_knowledge.txt",encoding='utf-8')as f:
    d = f.read()
#print(d)
data = re.split(r"\n\d+\.\s+",d)
#print(data[0,1])
data = [i.strip() for i in data if i.strip()]
print(len(data))
embedder = SentenceTransformer('all-miniLM-L6-v2')
vectors = []
for i, val in enumerate(data):
    embedded =  embedder.encode(val).tolist()
    vectors.append({"id": f"val-{i}","values":embedded, "metadata":{"text":val,"source":"company_knowledge.txt","type":"policy"}})
batch_s = 50
for i in range(0, len(vectors), batch_s):
    index.upsert(vectors = vectors[i:i+batch_s])


                    







