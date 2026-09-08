import streamlit as st

lab1 = st.Page(
    "Lab1.py",
    title="Lab 1",
    icon="1️⃣",
)

lab2 = st.Page(
    "Lab2.py",
    title="Lab 2",
    icon="2️⃣",
    default=True,
)

lab3 = st.Page(
    "Lab3.py",
    title="Lab 3",
    icon="3️⃣",
)

selected_page = st.navigation([lab1, lab2, lab3])
selected_page.run()