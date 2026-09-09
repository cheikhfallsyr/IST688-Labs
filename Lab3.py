import streamlit as st
from openai import OpenAI


st.title("Lab 3: Streaming Chatbot with Memory")

system_message = {
    "role": "system",
    "content": (
        "You are a helpful question-answering chatbot. Explain every answer "
        "in simple language that a 10-year-old can understand. When the user "
        "asks a new question, answer it and end with exactly: Do you want more "
        "info? If the user answers yes, provide more information about the "
        "same topic and end with exactly: Do you want more info? If the user "
        "answers no, reply with exactly: How can I help you?"
    ),
}
if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get the user's next message.
if prompt := st.chat_input("Type your message here"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    user_message_indexes = [
        index
        for index, message in enumerate(st.session_state.messages)
        if message["role"] == "user"
    ]

    buffer_start = (
        user_message_indexes[-2]
        if len(user_message_indexes) >= 2
        else user_message_indexes[0]
    )

    conversation_buffer = st.session_state.messages[buffer_start:]
    messages_for_model = [system_message] + conversation_buffer

    # Stream the answer from OpenAI.
    client = st.session_state.client
    stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages_for_model,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )