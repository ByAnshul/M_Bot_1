from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot-try"
embeddings = download_hugging_face_embeddings()

# Check if index exists first
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print("Already Created")


docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# Load PDF data
raw_text = load_pdf_file("NewData/")
if not raw_text:
    print("❌ No text found! Check your 'Data/' folder and 'load_pdf_file()' function.")
else:
    print(f"✅ Loaded Text: {raw_text[:500]}")  # Print first 500 characters

    # Split text into chunks
    text_chunks = text_split(raw_text)
    print(f"✅ Total Chunks Created: {len(text_chunks)}")
    
    if text_chunks:
        print("Uploading to Pinecone...")
        docsearch.add_documents(documents=text_chunks)
        print("✅ Upload completed!")
    else:
        print("❌ No text chunks created. Check 'text_split()' function.")