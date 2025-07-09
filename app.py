import streamlit as st
from agent import handle_query

st.set_page_config(page_title="Zomato-GenAI Agent", layout="centered")
st.title("🍜 GenAI Restaurant Agent")
q = st.text_input("Ask me anything about menu or order...")

if st.button("Ask") and q:
    with st.spinner("Thinking..."):
        st.write(handle_query(q))