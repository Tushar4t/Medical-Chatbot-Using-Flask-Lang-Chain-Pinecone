from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def download_hugging_face_embeddings():
    # Runs all-MiniLM-L6-v2 locally (384-d), so no API token or credits needed
    return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")