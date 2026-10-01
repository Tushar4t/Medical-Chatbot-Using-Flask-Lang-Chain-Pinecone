import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    split_docs,
    download_hugging_face_embeddings
)

load_dotenv()

# Load API Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in environment variables.")

os.environ["PINECONE_API_KEY"] = pinecone_api_key

# 1. Process documents
print("Loading PDF files...")
extracted_data = load_pdf_file("data")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = split_docs(filtered_data)

print(f"Total PDF pages loaded: {len(extracted_data)}")
print(f"Total text chunks created: {len(text_chunks)}")

if len(text_chunks) == 0:
    raise ValueError("No text chunks generated! Verify 'Medical_book.pdf' exists in 'data/'.")

# 2. Get embeddings model
embeddings = download_hugging_face_embeddings()

# 3. Initialize Pinecone Client
pc = Pinecone(api_key=pinecone_api_key)
index_name = "medical-chatbot"

# Check if index exists using the correct SDK attribute (.name)
existing_indexes = [i.name for i in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the index
index = pc.Index(index_name)

# 4. Upsert documents into Pinecone vector store
print("Upserting vectors to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("Indexing complete! Vector store successfully populated.")