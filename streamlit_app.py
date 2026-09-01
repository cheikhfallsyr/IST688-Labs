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

selected_page = st.navigation([lab1, lab2])
selected_page.run()