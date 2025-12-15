from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

path_root = os.getcwd()
path = os.path.abspath(
    os.path.join(path_root, 'data_base', 'REhsani_DataScience_1p_v2.pdf')
)

CHROMA_DIR = os.path.abspath(
    os.path.join(path_root, 'data_base', 'chroma_pdf_db')
)


loader = PyPDFLoader(path)
documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max characters per chunk (roughly)
    chunk_overlap=200,    # overlap between chunks to keep context
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

for i, chunk in enumerate(chunks[:3]):
    print("-" * 80)
    print(f"Chunk {i}")
    print(f"Metadata: {chunk.metadata}")
    print(chunk.page_content[:], "...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,   # folder on disk
)

vectorstore.persist()