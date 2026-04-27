import streamlit as st

st.set_page_config(page_title="Translation", layout="wide")
st.title("NLP Translation")

pg = st.navigation([
    st.Page("language_embeddings.py", title="Language Embeddings"),
    st.Page("precomputed_language_embeddings.py", title="Precomputed Language Embeddings"),
    st.Page("sentence_embeddings.py", title="Sentence Embeddings"),
    st.Page("sparse_autoencoders.py", title="Sparse Autoencoders"),
    st.Page("idioms.py", title="Idioms"),
])

st.sidebar.title("Embedding Explorer")

pg.run()
