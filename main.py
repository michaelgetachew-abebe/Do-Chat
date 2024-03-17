import streamlit as st 
import pandas as pd 
import numpy as np 
import json

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from llama_cpp import Llama
import requests

st.title("DoChat 💬       *Chat with your Document*")

pdf = st.file_uploader("Upload your PDF Document", type="pdf")

def document_processor(doc):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(doc)

    emb = HuggingFaceEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, emb)

    return knowledgeBase

if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    knowledgeBase = document_processor(text)

    query = st.text_input('Ask a question to the PDF')
    cancel_button = st.button('Cancel')

    if cancel_button:
        st.stop()

    if query:
        docs = knowledgeBase.similarity_search(query)
        import json
        
        system_message = "You are a helpful assistant"
        user_message = query
        context = docs
        max_tokens = 3000

        prompt = f"""<s>[INST] <<SYS>>
                {system_message}
                <</SYS>>
                Context: {context}
                Question: {user_message}
                Answer:
            [/INST]"""
        
        model_path = './llama-2-7b-chat.Q2_K.gguf'
        model = Llama(model_path=model_path)

        output = model(prompt, max_tokens=max_tokens, echo=True)

        st.write(output)