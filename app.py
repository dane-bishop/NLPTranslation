import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("Mini Multilingual Embedding Demo")

lang1 = st.selectbox("First language", ["English", "Spanish", "German"])
lang2 = st.selectbox("Second language", ["English", "Spanish", "German"], index=1)
model = st.selectbox("Embedding model", ["fastText", "mBERT", "LaBSE"])

sentence1 = st.text_input("Sentence 1", "The cat sits on the mat.")
sentence2 = st.text_input("Sentence 2", "El gato se sienta en la alfombra.")

def fake_embed(text: str, dim: int = 8):
    rng = np.random.default_rng(abs(hash((text, model))) % (2**32))
    return rng.normal(size=(dim,))

vec1 = fake_embed(sentence1)
vec2 = fake_embed(sentence2)

sim = cosine_similarity([vec1], [vec2])[0, 0]

st.metric("Cosine similarity", f"{sim:.3f}")

neighbors = pd.DataFrame({
    "candidate": [f"{lang2}_example_{i}" for i in range(5)],
    "similarity": np.sort(np.random.rand(5))[::-1]
})

st.subheader("Nearest neighbors")
st.dataframe(neighbors)