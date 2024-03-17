FROM python:3.9-slim

WORKDIR /app

RUN apt update && apt install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./llama-2-7b-chat.Q2_K.gguf ./app/llama-2-7b-chat.Q2_K.gguf
COPY ./main.py ./app/main.py

RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org llama-cpp-python
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org streamlit
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org langchain
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org PyPDF2
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org sentence-transformers
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org faiss-cpu

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "./app/main.py", "--server.port=8501"]