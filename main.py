import os
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint



def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        task="text-generation",
        max_new_tokens=100,
        stop_sequences=['\n\n', '\n\nExplanation', "\n"],
        huggingfacehub_api_token="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")
    return llm


def load_db():
    data = []
    for file in os.listdir("knowledge-base"):
      if file.endswith(".csv"):
        loader = CSVLoader(file_path="knowledge-base/"+file, encoding="utf-8", csv_args={
                    'delimiter': ','})
        if data:
            data.extend(loader.load())
        else:
            data = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local("db")

def conversational_chat(query):
    try:
        result = chain({"question": query, 'chat_history': []})
        print(result)
        return result["answer"]
    except Exception as e:
        print(e)
        return "I Dont Know!!!Please Reframe your Question"
llm = load_llm()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})
db = FAISS.load_local("db", embeddings, allow_dangerous_deserialization=True)
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Please write your query here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        output = conversational_chat(prompt)
        response = st.write(output)
    st.session_state.messages.append({"role": "assistant", "content": output})


##Uncomment below line to load data
# load_db()
# conversational_chat("What is country code of india?")
