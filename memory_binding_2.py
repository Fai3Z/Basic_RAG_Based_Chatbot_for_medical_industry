# this is latest approach to load model from huggigface. however model safetensor files load everytime program is run (large GB files) so inference takes way too long. better to use pipeline

import os  # Load the HF token for gated model access
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # Load environment variables

# Load the tokenizer and model directly
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def generate_response(prompt, max_length=384, temperature=0.5):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of relevant information provided in the context to answer the user's question.
If the answer is not present in your knowledge base, just say that you don't know the answer, and don't try to make up an answer.
Don't provide anything out of the given context. Stick to the context as much as possible.

Context: {context}
Question: {question} 

Start the answer directly. No small talk-based content, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load the database using FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the QA chain
def create_qa_chain():
    return RetrievalQA.from_chain_type(
        llm=generate_response,  # Use the direct response generator function
        chain_type="stuff",  # Define which chain is used
        retriever=db.as_retriever(search_kwargs={'k': 3}),  # Retrieve relevant results
        return_source_documents=True,  # Include source metadata
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

qa_chain = create_qa_chain()

# Invoke the chain with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])


# access to model is restricted
