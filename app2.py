from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
import subprocess

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt  # Assuming system_prompt is defined

# Initialize FastAPI
app = FastAPI()

# Load environment variables
load_dotenv()

# Retrieve API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Ensure keys are available for libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

# Load the Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static files (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- ROUTES ----------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/main", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get", response_class=JSONResponse)
async def get_response(msg: str = Form(...)):
    print("User input:", msg)
    
    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]
    
    print("Response:", answer)
    return {"response": answer}

# Route to start the FastAPI app as a subprocess
@app.get("/start-app")
async def start_app():
    subprocess.Popen(["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8080", "--reload"], shell=True)
    return {"message": "App started"}

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
