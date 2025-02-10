print("ds")
# print(%pwd)
import os 
os.chdir("../")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents
os.makedirs("Data", exist_ok=True)
print(os.path.abspath("Data/"))

extracted_data = load_pdf_file(data="D:/Major Project/YT bot 2/Data")
print("bfkjbkwejb")
print(extracted_data)
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "D:\Major Project\YT bot 2\Medical-Bot\Data\Book 3 edition.pdf"  # Replace with an actual file
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(documents)
