import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# For image processing
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot-try"
embeddings = download_hugging_face_embeddings()

# Check if index exists first
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimension for text embeddings from your Hugging Face model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print("Already Created")

# Create vector store for text-based documents
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

# ---------------------------
# Uploading PDF/Text Data
# ---------------------------
raw_text = load_pdf_file("NewData/")  # Make sure your PDF extraction function works
if not raw_text:
    print("❌ No text found! Check your 'NewData/' folder and 'load_pdf_file()' function.")
else:
    print(f"✅ Loaded Text: {raw_text[:500]}")  # Print first 500 characters

    # Split text into chunks
    text_chunks = text_split(raw_text)
    print(f"✅ Total Chunks Created: {len(text_chunks)}")
    
    if text_chunks:
        print("Uploading text chunks to Pinecone...")
        docsearch.add_documents(documents=text_chunks)
        print("✅ Text upload completed!")
    else:
        print("❌ No text chunks created. Check 'text_split()' function.")

# ---------------------------
# Uploading Image Data (e.g., X-rays)
# ---------------------------

# Initialize CLIP model and processor for image embeddings
clip_model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

def get_image_embedding(image_path):
    """
    Generate an embedding vector for an image using CLIP.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    # Normalize the vector (optional but recommended)
    embedding = outputs[0] / outputs[0].norm()
    return embedding.squeeze().tolist()  # Convert to list for Pinecone

def upload_image_embeddings(image_folder, pinecone_index):
    """
    Processes all images in the given folder and uploads their embeddings to Pinecone.
    """
    # List image files (jpg and png)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("❌ No image files found in the folder:", image_folder)
        return

    vectors = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            embedding = get_image_embedding(image_path)
            vector_id = f"image-{os.path.splitext(image_file)[0]}"  # Unique ID for the image
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {"source": image_path, "type": "xray"}
            })
            print(f"✅ Processed {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    # Upsert vectors into the Pinecone index
    if vectors:
        # Using the Pinecone GRPC client to upsert directly
        index = pc.Index(index_name)
        index.upsert(vectors=vectors)
        print("✅ Image upload completed!")
    else:
        print("❌ No vectors to upload.")

# Folder containing X-ray images (e.g., "NewData/xrays/")
upload_image_embeddings("NewData/xrays/", index_name)
