# here we will set up the LLM i.e mistral via HF space and connect the mistral LLM with FAISS and create chain
import os # (load the HF token for gated model access on HF spaces)
# import huggigface endpoint
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"384"}
    )
    return llm

# endpoint = HuggingFaceEndpoint( # testing if HF_token auth was successful and if LLM is responding?
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="text-generation",
#     model_kwargs={"token": HF_TOKEN, "max_length": 50}
# )
# response = endpoint("This is a test prompt.")
# print(response)

# connecting LLM with FAISS and creating chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of relevant information provided in the context to answer user's question.
If the answer is not present in your knowledge base, just say that you dont know the answer, and dont try to make up an answer. 
Dont provide anything out of the given context. stick to context as much as possible

Context: {context}
Question: {question} 

Start the answer directly. No small talk based content please.
"""
# we have defined the template
# context is data retrieved by RAG from knowledge base
# question is question asked by user.

# this template will be passed to the LLM to provide relevant answer

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# loading the database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True) # set to true when we trust the source (knowledge base)


# creating the Question-Answer chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID), # which LLM used
    chain_type="stuff",  # which chain used?
    retriever=db.as_retriever(search_kwargs={'k':3}), # where is data being fetched from (how many similar documents or chunks retrieve from based on question from knowledge base. we set 3 so 3 relevant results will be retrieved based on which answer will be generated. (the search will be semantic search from pdf or group of pdfs based on best result (eucleadian distance)))
    return_source_documents=True, # metadata i.e which page source data retrieved from
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}

)
# now chain is setup

# now we will invoke (activate) chain with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
