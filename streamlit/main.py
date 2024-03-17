import streamlit as st 
import pandas as pd 
import numpy as np 
import json
import tempfile

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llama_cpp import Llama
import requests

def app():
    st.title("DoChat 💬       *Chat with your Document/CSV*")

    upload_file = st.file_uploader("Upload your PDF Document")

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
    
    def csv_processor(csv_data):
        emb = HuggingFaceEmbeddings()
        knowledgeBase = FAISS.from_documents(csv_data, emb)

        return knowledgeBase

    if upload_file is not None:
        if str(upload_file.name)[-4:] == ".pdf":
            pdf_reader = PdfReader(upload_file)

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
                max_tokens = 512

                prompt = f"""<s>[INST] <<SYS>>
                    {system_message}
                    <</SYS>>
                    Context: {context}
                    Question: {user_message}
                    Answer:
                [/INST]"""
            
                model_path = './app/llama-2-7b-chat.Q2_K.gguf'
                model = Llama(model_path=model_path, n_ctx=4096)

                output = model(prompt, max_tokens=max_tokens, echo=True)

                st.write(output["choices"][0]["text"][output["choices"][0]["text"].find("Answer:"):])
            
        else:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(upload_file.getvalue())
                temp_file_path = temp_file.name
                
            loader = CSVLoader(file_path = temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
            data = loader.load()

            st.write(data)

            knowledgeBase = csv_processor(data)

            query = st.text_input('Ask a question to the CSV')
            cancel_button = st.button('Cancel')

            if cancel_button:
                st.stop()

            if query:
                docs = knowledgeBase.similarity_search(query)
                import json
            
                system_message = "You are a helpful assistant"
                user_message = query
                context = docs
                max_tokens = 512

                prompt = f"""<s>[INST] <<SYS>>
                    {system_message}
                    <</SYS>>
                    Context: {context}
                    Question: {user_message}
                    Answer:
                [/INST]"""
            
                model_path = './app/llama-2-7b-chat.Q2_K.gguf'
                model = Llama(model_path=model_path, n_ctx=4096)

                output = model(prompt, max_tokens=max_tokens, echo=True)

                st.write(output["choices"][0]["text"][output["choices"][0]["text"].find("Answer:"):])

if __name__ == "__main__":
    app()