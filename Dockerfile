FROM python

WORKDIR /app

COPY ./llama-2-7b-chat.Q2_K.gguf ./app/llama-2-7b-chat.Q2_K.gguf
COPY ./app.py /app/app.py

RUN pip install llama-cpp-python
RUN pip install Flask

EXPOSE 5000

CMD ["python", "app.py"]