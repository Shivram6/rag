import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()
API = os.getenv("api")
pc = Pinecone(api_key=API)

index_name = "rag-index"

existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        
            region="us-east-1"  
        )
    )
    print("Index created successfully!")
else:
    print("Index already exists!")