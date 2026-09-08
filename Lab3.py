import streamlit as st
from openai import OpenAI


st.title("Lab 3: Streaming Chatbot with Memory")

if "client" not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

# Display the conversation.
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

    # Keep the last two user messages and their assistant responses.
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

    # Stream the answer from OpenAI.
    client = st.session_state.client
    stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=conversation_buffer,
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )