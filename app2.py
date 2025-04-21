from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import os
from dotenv import load_dotenv
import subprocess
from werkzeug.utils import secure_filename
import tempfile
import uuid
from collections import deque
from datetime import datetime
import base64
import requests
from places import MedicalPlacesSystem  # Import for medical places recommendations

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import get_system_prompt, customize_response
from src.database import get_user_health, create_user, verify_user, init_db

# Initialize the database
init_db()

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

# Initialize Pinecone index
index_name = "medicalbot-try"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize LLM
llm = ChatOpenAI(
    openai_api_key=TOGETHER_API_KEY,
    openai_api_base="https://api.together.xyz/v1",
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    temperature=0.4,
    max_tokens=500
)

# Note: Global prompt chain initialization is removed.
# We will rebuild the prompt chain dynamically within the /get route.

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store uploaded document IDs
uploaded_docs = {}

# Initialize MedicalPlacesSystem
print("Initializing MedicalPlacesSystem...")
global medical_system
try:
    medical_system = MedicalPlacesSystem()
    print("MedicalPlacesSystem initialized successfully")
except Exception as e:
    print(f"Error initializing MedicalPlacesSystem: {str(e)}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    medical_system = None  # Explicitly set to None on failure

def init_session():
    """Initialize session variables if they don't exist"""
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    if 'current_context' not in session:
        session['current_context'] = {}

def format_conversation_history():
    """Format conversation history for context"""
    if 'conversation_history' not in session:
        return ""
    
    formatted_history = []
    for exchange in session['conversation_history'][-5:]:  # Get last 5 exchanges
        formatted_history.append(f"User: {exchange['user']}")
        formatted_history.append(f"Assistant: {exchange['assistant']}")
    return "\n".join(formatted_history)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    """Process PDF file and return text chunks"""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(pages)
    return chunks

# ---------- ROUTES ----------

@app.route('/')
def index():
    return render_template('index.html')  # Homepage

@app.route('/chat')
def chat():
    print("Current session data:", dict(session))  # Debug log
    
    # Allow both logged-in users and guests
    if 'user_id' not in session:
        print("No user_id in session, redirecting to signin")  # Debug log
        return redirect(url_for('signin'))
        
    return render_template('chat.html')

@app.route('/main')
def main():
    # Redirect to signin if not logged in
    if 'user_id' not in session:
        return redirect(url_for('signin'))
    return render_template('chat.html')

@app.get("/signin")
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
    try:
        data = request.get_json()
        print("Login attempt for email:", data['email'])  # Debug log
            
        user = verify_user(data['email'], data['password'])
        
        if user:
            session.clear()
            # Set session data
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_name'] = user['name']
            session['is_guest'] = False
            session.modified = True
                
            print("Login successful - Session data:", dict(session))  # Debug log
                
            return jsonify({
                'success': True,
                'redirect': '/chat',
                'user_name': user['name']
            })
        else:
            print("Login failed - Invalid credentials")  # Debug log
            return jsonify({
                'success': False,
                'message': 'Invalid email or password'
            })
    except Exception as e:
        print("Login error:", str(e))  # Debug log
        return jsonify({
            'success': False,
            'message': 'An error occurred during login'
        }), 500

@app.route('/guest_login', methods=['POST'])
def guest_login():
    """Handle guest login by setting a guest session"""
    try:
        session.clear()
        session['user_id'] = 'guest'
        session['user_email'] = 'guest@example.com'
        session['user_name'] = 'Guest'
        session['is_guest'] = True
        session.modified = True
        
        print("Guest login successful - Session data:", dict(session))  # Debug log
        
        return jsonify({
            'success': True,
            'redirect': '/chat'
        })
    except Exception as e:
        print("Guest login error:", str(e))  # Debug log
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/get', methods=["POST"])
def get_response():
    msg = request.form["msg"]
    print("User input:", msg)
    
    try:
        init_session()
        context = format_conversation_history()
        
        # Get user's health information (if not guest)
        health_info = None
        if 'user_id' in session and session['user_id'] != 'guest':
            health_info = get_user_health(session['user_id'])
        
        # Rebuild the system prompt with updated context using a placeholder for {context}
        system_prompt_template = get_system_prompt() + "\n\nPrevious conversation context:\n{context}"
        print("System prompt template:", system_prompt_template)  # Debug log
        
        # Build the prompt chain with both 'input' and 'context' placeholders
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("human", "{input}")
        ])
        
        # Recreate the chain with the updated prompt
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Invoke the chain with both variables
        response = rag_chain.invoke({"input": msg, "context": context})
        print("Raw response from rag_chain.invoke:", response)  # Debug log
        
        answer = response.get("answer")
        if answer is None:
            raise Exception("No 'answer' key found in response: " + str(response))
        
        # Customize response based on user's health information
        if health_info:
            answer = customize_response(
                answer,
                symptoms=health_info.get('symptoms'),
                diseases=health_info.get('diseases')
            )
        
        print("Final Response:", answer)
        session['conversation_history'].append({
            "user": msg,
            "assistant": answer
        })
        if len(session['conversation_history']) > 5:
            session['conversation_history'] = session['conversation_history'][-5:]
        session['current_context'] = {
            "last_question": msg,
            "last_answer": answer
        }
        session.modified = True
        
        return str(answer)
    except Exception as e:
        print("Error during get_response:", str(e))
        return "I apologize, but I encountered an error while processing your question. Please try again."

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        print("Error: No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        print("Error: No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        print(f"Error: Invalid file type for {file.filename}")
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400

    try:
        doc_id = str(uuid.uuid4())
        print(f"Processing file: {file.filename} with doc_id: {doc_id}")
        
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)
        print(f"File saved temporarily at: {file_path}")

        print("Processing PDF...")
        chunks = process_pdf(file_path)
        print(f"Generated {len(chunks)} chunks from PDF")
        
        print("Adding chunks to Pinecone...")
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'doc_id': doc_id,
                'chunk_index': i,
                'filename': filename,
                'upload_time': datetime.now().isoformat()
            })
        
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            docsearch.add_documents(documents=batch)
            print(f"Added batch {i//batch_size + 1} to Pinecone")
        
        uploaded_docs[doc_id] = {
            'filename': filename,
            'chunks': len(chunks),
            'upload_time': datetime.now().isoformat()
        }
        
        print(f"Successfully processed and stored {filename}")
        
        os.remove(file_path)
        os.rmdir(temp_dir)

        return jsonify({
            'message': 'File processed and stored successfully',
            'chunks': len(chunks),
            'doc_id': doc_id
        }), 200
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        try:
            if 'file_path' in locals():
                os.remove(file_path)
            if 'temp_dir' in locals():
                os.rmdir(temp_dir)
        except:
            pass
        return jsonify({'error': str(e)}), 500

@app.route("/delete_document", methods=["POST"])
def delete_document():
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        
        if not doc_id:
            return jsonify({'error': 'No document ID provided'}), 400
            
        if doc_id not in uploaded_docs:
            return jsonify({'error': 'Document not found'}), 404
            
        docsearch.delete(
            filter={
                "doc_id": doc_id
            }
        )
        
        del uploaded_docs[doc_id]
        
        return jsonify({
            'message': 'Document deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/cleanup_session", methods=["POST"])
def cleanup_session():
    try:
        data = request.get_json()
        doc_ids = data.get('doc_ids', [])
        
        for doc_id in doc_ids:
            if doc_id in uploaded_docs:
                docsearch.delete(
                    filter={
                        "doc_id": doc_id
                    }
                )
                del uploaded_docs[doc_id]
        
        return jsonify({
            'message': 'Session cleaned up successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/get_summary", methods=["POST"])
def get_summary():
    try:
        data = request.get_json()
        doc_id = data.get('doc_id')
        
        if not doc_id:
            return jsonify({'error': 'No document ID provided'}), 400
            
        print(f"Generating summary for doc_id: {doc_id}")
        
        results = docsearch.similarity_search(
            query="",
            k=5,
            filter={"doc_id": doc_id}
        )
        
        if not results:
            print(f"No documents found for doc_id: {doc_id}")
            return jsonify({'summary': 'No content found in document'}), 200
            
        print(f"Found {len(results)} chunks for summarization")
        
        content = "\n".join([doc.page_content for doc in results])
        
        if any(keyword in content.lower() for keyword in ['covid', 'sars-cov-2', 'rt-pcr']):
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Patient name, test date, result (positive/negative), and key recommendations in 2-3 lines."),
                ("human", "Analyze this COVID-19 report:\n\n{content}")
            ])
        elif any(keyword in content.lower() for keyword in ['cbc', 'hemoglobin', 'wbc', 'rbc', 'platelets']):
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Patient name, test date, and any abnormal values that need attention in 2-3 lines."),
                ("human", "Analyze this blood test report:\n\n{content}")
            ])
        elif any(keyword in content.lower() for keyword in ['prescription', 'rx', 'tablet', 'capsule', 'mg']):
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Key medications prescribed and their primary purpose in 2-3 lines."),
                ("human", "Analyze this prescription:\n\n{content}")
            ])
        else:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Main content and purpose in 1-2 lines."),
                ("human", "Analyze this medical document:\n\n{content}")
            ])
        
        summary_chain = summary_prompt | llm
        summary_response = summary_chain.invoke({
            "content": content
        })
        
        print(f"Generated medical analysis: {summary_response.content}")
        
        return jsonify({
            'summary': summary_response.content
        }), 200
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({
            'error': 'Failed to generate summary'
        }), 500

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    try:
        audio_data = request.json['audio_data']
        audio_content = base64.b64decode(audio_data)
        audio_base64 = base64.b64encode(audio_content).decode("utf-8")

        url = f"https://speech.googleapis.com/v1/speech:recognize?key={os.environ.get('GOOGLE_API_KEY', 'AIzaSyAcyY1XoBCNg8qcSYk9oDeChC40-PzkevA')}"
        headers = {"Content-Type": "application/json"}
        data = {
            "config": {
                "encoding": "WEBM_OPUS",
                "sampleRateHertz": 48000,
                "languageCode": "en-US"
            },
            "audio": {
                "content": audio_base64
            }
        }

        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if "results" in response_data:
            text = response_data['results'][0]['alternatives'][0]['transcript']
            return jsonify({"text": text})
        else:
            return jsonify({"error": response_data.get('error', {}).get('message', 'Error processing audio')})
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/find_medical_help', methods=['POST'])
def find_medical_help():
    try:
        data = request.get_json()
        disease = data.get('disease')
        location = data.get('location')
        
        print(f"Received request for medical help - Disease: {disease}, Location: {location}")
        
        if not disease or not location:
            print("Missing disease or location in request")
            return jsonify({
                'success': False,
                'message': 'Please provide both disease and location'
            }), 400
            
        print("Calling medical_system.get_recommendations()")
        try:
            if not medical_system:
                raise Exception("Medical recommendation system not initialized")
            result = medical_system.get_recommendations(disease, location)
            print(f"Result from medical_system: {result}")
        except Exception as e:
            print(f"Error in medical_system.get_recommendations(): {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error getting recommendations: {str(e)}'
            }), 500
        
        if not result['success']:
            print(f"Failed to get recommendations: {result.get('message')}")
            return jsonify(result), 404
            
        response = result['response']
        print("Successfully formatted response")
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        print(f"Error in find_medical_help: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'An error occurred while finding medical help: {str(e)}'
        }), 500

@app.route('/start-app')
def start_app():
    subprocess.Popen(["python", "app.py"], shell=True)
    return "App started", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
