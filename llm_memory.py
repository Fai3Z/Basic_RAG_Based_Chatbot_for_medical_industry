# we will load the pdf file required as knowledge base for our 
# project. then we will create chunks of the attachments for context 
# retrieval by RAG system. Followed by embedding creation for those 
# chunks and storing tose embeddings in a vecto DB like FAISS or weaviate

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # to embed the text chunks for the RAG based semantic search (with help of sentence-transformers/allMiniLM-L6-v2)
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# loading the pdf file from directory "/data"
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print(len(documents))

# creating the chunks using charactertextsplitter using the langchain library and deciding chunk size and chunk overla across different pages
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))


# now to create vector embeddings
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

#  storing the vector embeddings in a VectorDB
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model) # vector db containing chunks using embedding model stored in FAISS DB
db.save_local(DB_FAISS_PATH)




