# Basic_RAG_Based_Chatbot_for_medical_industry
A basic chatbot built using langchain and streamlit that can help with medical diagnosis with the help of medical attachments as context

### Detail
This is amedical chatbot that uses a medical encyclopedia pdf source to provide answers to a users queries. It is a basic RAG (Retrieval Augmented Generation) chatbot that has been tuned to provide sources with the answers as well. The chatbot also contains memory retention of chats per session. This is done by using session state variables as a list and appending them (messages) for the user running the application. The interface is developed on streamlit

### Using the project
#### 1. Clone the repository
in your editors terminal in the desired directory, type:
```sh
git clone https://github.com/Fai3Z/Basic_RAG_Based_Chatbot_for_medical_industry.git
```

and install requirement packages
```sh
pip install -r requirements.txt
```

#### 2. Create the API token for huggingface model inference
First create a virtual environment in the same directory where project files were cloned using (for VS code):
```sh
python -m venv env
env\Scripts\activate

Now create  huggingface account and get a HF_Token. Make a ".env" file in the same directory containing "requirements.txt" and paste your token variable there for the models to call and reference.
i.e
.env (secret tokens/keys file)
 HF_TOKEN="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"
```

#### 3. Running Code files
run the following files:
1. llm_memory.py (To divide the attachment source (medical encyclopedia into LLM's chunks))
2. memory_binding_with_llm.py (loads vector DB in a directory and vectorizes text embeddings, initializes and invokes the chains for the LLM)
3. ragmedbot.py (calls all relevant functions and serves the application on streamlit).

##### Note: to run the application on streamlit run
```sh
streamlit run ragmedbot.py
```
This will serve the chatbot application on http://localhost:8501/

#### Expected Output
![Description](https://github.com/Fai3Z/Basic_RAG_Based_Chatbot_for_medical_industry/blob/main/output1.png?raw=true)
![Description](https://github.com/Fai3Z/Basic_RAG_Based_Chatbot_for_medical_industry/blob/main/output2.png?raw=true)



#### 4. Add more sources and improve upon the project with more features, wider data sources and better models!
