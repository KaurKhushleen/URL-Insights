import os
import streamlit as st
import pickle
import time
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from myAPI import api_key

FILE_PATH = "vec_embeddings.pkl"
llm = GoogleGenerativeAI(model = "gemini-2.0-flash", api_key= api_key)
st.title("URL LLM")

st.sidebar.title("sidebar title")

urls = []

for i in range(1,4):
    url = st.text_input(f"URL {i}")
    urls.append(url)


process_url_button = st.button("Process URLs")

status_placeholder = st.empty()
progress_placeholder = st.empty()

if process_url_button:

    #loading data
    status_placeholder.text("loading data")
    progress_placeholder.progress(0)
    loader = UnstructuredURLLoader(urls = urls)
    progress_placeholder.progress(10)
    data = loader.load()
    progress_placeholder.progress(20)
    time.sleep(0.1)

    #splitting data
    status_placeholder.text("splitting data")
    progress_placeholder.progress(30)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    
    chunks = splitter.split_documents(data)
    progress_placeholder.progress(40)
    time.sleep(2)

    #create embeddings
    status_placeholder.text("creating embeddings")
    progress_placeholder.progress(50)
    embeddings = SentenceTransformerEmbeddings(model_name = "all-mpnet-base-v2")
    vectors = FAISS.from_documents(chunks, embeddings)
    progress_placeholder.progress(70)


    # saving embeddings in file
    status_placeholder.text("saving embeddings")
    time.sleep(1)
    progress_placeholder.progress(80)
    with open(FILE_PATH, "wb") as f:
       pickle.dump(vectors, f)
    
    status_placeholder.text("Embeddings Saved✅")
    progress_placeholder.progress(100)
    status_placeholder.empty()
    progress_placeholder.empty()


query = st.text_input("Enter query: ")
if os.path.exists(FILE_PATH):
    with open(FILE_PATH, "rb") as f:
        vector_index = pickle.load(f)

process_query = st.button("Process Query")
chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vector_index.as_retriever())

if process_query:
    response = chain.invoke({"question" : query})
    st.write(response["answer"])
    st.write(response["sources"])
