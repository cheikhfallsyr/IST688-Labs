# Lab2.py
import streamlit as st
from openai import OpenAI
st.title("Lab 2")
# Show title and description.
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    # Create an OpenAI client.
oopenai_api_key = st.secrets["OPENAI_API_KEY"]

try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"]
    )
    client.models.list()
except Exception as error:
    st.error(f"{type(error).__name__}: {error}")
    st.stop()

# This must be outside the except block.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)",
    type=("txt", "md"),
)
    # Ask the user for a question via `st.text_area`.
question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)