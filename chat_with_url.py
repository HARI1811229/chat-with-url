import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHubLL

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


def main():

    st.title('Chat With Website')
    st.subheader('Input your website URL, ask questions, and receive answers directly from the website.')

    url = st.text_input("Insert The website URL")

    prompt = st.text_input("Ask a question (query/prompt)")
    if st.button("Submit Query", type="primary"):
        ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
        DB_DIR: str = os.path.join(ABS_PATH, "db")

        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100)

        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        vectordb = Chroma.from_documents(documents=docs,
                                         embedding=embeddings,
                                         persist_directory=DB_DIR)

        vectordb.persist()


        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = RetrievalQA.from_chain_type(llm= HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}), chain_type="stuff", retriever=retriever,memory = memory)

        response = qa(prompt)
        st.write(response)


if __name__ == '__main__':
    main()

#streamlit run D:\Hari\multi_agent\venv\chat_with_url.py  use this comman in command prompt to run the application

