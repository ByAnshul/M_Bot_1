from flask import Flask, render_template, jsonify, request, redirect, url_for,session
import os
from dotenv import load_dotenv
import subprocess

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI

from src.helper import download_hugging_face_embeddings
from src.database import create_user, verify_user, init_db
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI with Together API
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt  # Assuming system_prompt is defined

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-123')

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')

# Ensure keys are available for libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot-try"

# Load the Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize LLM with Together API
llm = ChatOpenAI(
    openai_api_key=TOGETHER_API_KEY,
    openai_api_base="https://api.together.xyz/v1",
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.4,
    max_tokens=500
)

# Build prompt chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the question-answer and retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------- ROUTES ----------

@app.route('/')
def index():
    return render_template('index.html')  # Homepage

@app.route('/chat')
def chat():
    if 'user_email' not in session:
        return redirect(url_for('signin'))
    return render_template('chat.html')

@app.route('/main')
def main():
    return render_template('chat.html')  # Add main route



@app.get("/signin")  # Add this route
async def signin():
    return render_template("signin.html")




@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    success = create_user(
        data['name'],
        data['email'],
        data['password'],
        data.get('symptoms'),
        data.get('diseases')
    )
    
    return jsonify({
        'success': success,
        'message': 'User created successfully' if success else 'User already exists!'
    })

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = verify_user(data['email'], data['password'])
    
    if user:
        session['user_email'] = user['email']
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session.modified = True  # Ensure session changes are committed
        return jsonify({
            'success': True,
            'redirect': '/chat'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid email or password'
        })

@app.route('/get', methods=["POST"])
def get_response():
    msg = request.form["msg"]
    print("User input:", msg)
    
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    
    print("Response:", answer)
    return str(answer)

# Optional: Route to start the Flask app as a subprocess
@app.route('/start-app')
def start_app():
    subprocess.Popen(["python", "app.py"], shell=True)
    return "App started", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)