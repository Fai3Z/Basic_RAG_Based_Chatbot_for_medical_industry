import os
import streamlit as st

# Check and print package versions
print("os module: Built-in, no version")
print(f"streamlit: {st.__version__}") # 1.45.0

# LangChain modules
import langchain
import langchain_community
import langchain_core
import langchain_huggingface

print(f"langchain: {langchain.__version__}") # 0.3.24
print(f"langchain.community: {langchain_community.__version__}") # 0.3.23
#print(f"langchain_community.vectorstores: {community_vectorstores.__version__}")
print(f"langchain_core: {langchain_core.__version__}") # 0.3.56
print(f"langchain_huggingface: {langchain_huggingface.__version__}") # 0.1.2


# deploy on gradio if possible