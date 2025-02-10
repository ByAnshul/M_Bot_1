from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI with custom API base
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt  # assuming system_prompt is defined here
import os

app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')  # Use this for Together API

# Set the keys in os.environ (if needed for other libraries)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Get embeddings
embeddings = download_hugging_face_embeddings()

index_name = "medicalbot"

# Load the existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Instead of using the OpenAI wrapper, we now use ChatOpenAI with Together's API base.
llm = ChatOpenAI(
    openai_api_key=TOGETHER_API_KEY,  
    openai_api_base="https://api.together.xyz/v1",  # Together API endpoint
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.4,
    max_tokens=500
)

# Build your prompt chain (using your system prompt from src.prompt)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the question-answer chain and the overall retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask route for the chat interface
@app.route("/")
def index():
    return render_template('chat.html')

# Flask route for processing chat messages
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
