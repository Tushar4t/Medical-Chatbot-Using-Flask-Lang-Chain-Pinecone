import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def download_hugging_face_embeddings():
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing in environment variables!")

    # Uses the active Serverless Inference API endpoint
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token.strip()
    )
    return embeddings