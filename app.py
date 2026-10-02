import os
import uuid
from flask import Flask, render_template, jsonify, request, session
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

load_dotenv()

# Flask session needs a secret key to sign the cookie that stores chat history.
# Set SECRET_KEY in Render's environment variables for a stable key across restarts;
# falls back to a random key locally so it still works without one.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

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

# 3. Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY if GROQ_API_KEY else "missing_key",
    model_name="openai/gpt-oss-20b",
    temperature=0.4
)

# 4a. Contextualizer: rewrites a follow-up ("medicine for it") into a standalone
#     question ("medicine for headache") using the chat history, BEFORE retrieval runs.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 4b. Answer prompt: same rules as before, now with chat history available too.
system_prompt = (
    "You are a helpful medical assistant for answering questions.\n"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 4c. Full chain: rewrite question with history -> retrieve -> answer with history.
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# In-memory store of chat history, keyed by a per-visitor session id.
# NOTE: this resets whenever the Render free-tier instance spins down or restarts,
# and (on a multi-worker setup) is not shared across workers. Fine for a demo /
# single-worker deployment; swap for Redis or a DB for real multi-user persistence.
chat_histories = {}
MAX_TURNS = 6  # keep the last 6 exchanges so the prompt doesn't grow unbounded

@app.route("/")
def index():
    session["chat_id"] = str(uuid.uuid4())
    chat_histories[session["chat_id"]] = []
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY is missing in Render Environment variables.", 500

        msg = request.form["msg"]

        chat_id = session.get("chat_id")
        if not chat_id or chat_id not in chat_histories:
            chat_id = str(uuid.uuid4())
            session["chat_id"] = chat_id
            chat_histories[chat_id] = []

        history = chat_histories[chat_id]

        response = rag_chain.invoke({"input": msg, "chat_history": history})
        answer = response["answer"]

        history.append(HumanMessage(content=msg))
        history.append(AIMessage(content=answer))
        # trim to the last MAX_TURNS exchanges (2 messages per exchange)
        chat_histories[chat_id] = history[-(MAX_TURNS * 2):]

        return str(answer)
    except Exception as e:
        print(f"ERROR IN CHAT ENDPOINT: {e}")
        return f"Error processing request: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)