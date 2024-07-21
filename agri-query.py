# Agri Bot - Manas Dasgupta - 21st July 24

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def query(question, chat_history):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("agri_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Initialize a ConversationalRetrievalChain
    query = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=new_db.as_retriever(), 
        return_source_documents=True)
    # Invoke the Chain with
    return query({"question": question, "chat_history": chat_history})


def show_ui():
    st.title("Yours Truly Agri Bot")    
    st.image("teks-cbt.png")
    st.subheader("Please enter your Question below.")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your question here: "):
        # Invoke the function with the Retriver with chat history and display responses in chat container in question-answer pairs 
        with st.spinner("Working on your query...."):     
            response = query(question=prompt, chat_history=st.session_state.chat_history)            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])    

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

# Program Entry.....
if __name__ == "__main__":
    show_ui() 
    