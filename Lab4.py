import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from PyPDF2 import PdfReader


# Fix for using ChromaDB on Streamlit Community Cloud
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb


#### CREATE OPENAI CLIENT ####

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets.OPENAI_API_KEY
    )


#### ADD A DOCUMENT TO CHROMADB ####

def add_to_collection(collection, text, file_name):
    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
    )


#### EXTRACT TEXT FROM A PDF ####

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    return text


#### LOAD ALL PDFS INTO CHROMADB ####

def load_pdfs_to_collection(folder_path, collection):
    pdf_files = sorted(
        Path(folder_path).glob("*.pdf")
    )

    if len(pdf_files) != 7:
        st.error(
            "The Lab-04-Data folder must contain the seven PDF files."
        )
        st.stop()

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)

        add_to_collection(
            collection,
            text,
            pdf_file.name,
        )

    return len(pdf_files)


#### CREATE THE VECTOR DATABASE ####

def create_vector_database():
    chroma_client = chromadb.PersistentClient(
        path="./ChromaDB_for_Lab"
    )

    collection = (
        chroma_client.get_or_create_collection(
            "Lab4Collection"
        )
    )

    # Only embed the documents if the collection is empty
    if collection.count() == 0:
        load_pdfs_to_collection(
            "./Lab-04-Data/",
            collection,
        )

    st.session_state.Lab4_VectorDB = collection


# Only create or retrieve the database once per session
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner(
        "Creating the course vector database..."
    ):
        create_vector_database()
st.write(st.session_state.Lab4_VectorDB.get()["ids"])

#### MAIN APP ####

st.title("Lab 4: Chatbot using RAG")

st.write(
    "Ask a question about the seven course syllabi. "
    "The chatbot searches the ChromaDB collection "
    "and uses the three closest documents to answer."
)


#### LAB 3 CONVERSATION HISTORY ####

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you?",
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


#### GET THE USER'S NEXT MESSAGE ####

if prompt := st.chat_input(
    "Ask a question about the courses"
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    #### CREATE AN EMBEDDING FOR THE QUESTION ####

    client = st.session_state.openai_client

    response = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small",
    )

    query_embedding = response.data[0].embedding


    #### RETRIEVE THE THREE CLOSEST DOCUMENTS ####

    results = st.session_state.Lab4_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )


    #### COMBINE THE RETRIEVED DOCUMENTS ####

    rag_context = "\n\n---\n\n".join(
        (
            f"Source: {doc_id}\n\n"
            f"{document}"
        )
        for doc_id, document in zip(
            results["ids"][0],
            results["documents"][0],
        )
    )


    #### CREATE THE RAG SYSTEM PROMPT ####

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful course information chatbot. "
            "Answer the user's question using the retrieved "
            "course documents below. If the documents contain "
            "the answer, clearly say that your answer is based "
            "on the course documents and name the source file. "
            "If the answer is not in the documents, say that "
            "you could not find it in the course documents. "
            "Do not make up course information."
            "\n\nRetrieved course documents:\n\n"
            + rag_context
        ),
    }


    #### LAB 3 CONVERSATION BUFFER ####

    user_message_indexes = [
        index
        for index, message in enumerate(
            st.session_state.messages
        )
        if message["role"] == "user"
    ]

    buffer_start = (
        user_message_indexes[-2]
        if len(user_message_indexes) >= 2
        else user_message_indexes[0]
    )

    conversation_buffer = (
        st.session_state.messages[buffer_start:]
    )

    messages_for_model = [
        system_message
    ] + conversation_buffer


    #### GENERATE AND STREAM THE RESPONSE ####

    stream = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages_for_model,
        stream=True,
    )

    with st.chat_message("assistant"):
        answer = st.write_stream(stream)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )