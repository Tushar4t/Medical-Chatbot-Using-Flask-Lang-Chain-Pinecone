import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

load_dotenv()

# Safely read environment variables without crashing Gunicorn on startup
PINECONE_API_KEY = (os.environ.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY") or "").strip().strip("'").strip('"')
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or "").strip().strip("'").strip('"')

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 1. Download HuggingFace Embeddings
embeddings = download_hugging_face_embeddings()

# 2. Connect to Pinecone Vector Store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 3. Initialize Groq LLM using active openai/gpt-oss-20b model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY if GROQ_API_KEY else "missing_key",
    model_name="openai/gpt-oss-20b",   # <-- changed from llama-3.1-8b-instant
    temperature=0.4
)

# 4. Construct Prompt Template
system_prompt = (
    "You are a helpful medical assistant for answering questions.\n"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {input}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Define LCEL RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY is missing in Render Environment variables.", 500
            
        msg = request.form["msg"]
        response = rag_chain.invoke(msg)
        return str(response)
    except Exception as e:
        print(f"ERROR IN CHAT ENDPOINT: {e}")
        return f"Error processing request: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)