import streamlit as st

from webui.agentic_chat_page import agentic_chat_page


st.set_page_config(
    page_title="Agentic RAG Visualized",
    page_icon="🔎",
    layout="wide",
)


if __name__ == "__main__":
    agentic_chat_page()
