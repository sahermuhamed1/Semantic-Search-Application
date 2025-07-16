import streamlit as st
from sentence_transformers import SentenceTransformer

from load_embeddings import load_sentences, load_embeddings
from indexing import build_faiss_index, search

MODEL_NAME = "bert-base-nli-mean-tokens"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def main():
    st.title("Semantic Search with FAISS vector database")
    st.write("Choose a FAISS index type and search for similar sentences.")

    try:
        model = load_model()
        sentences = load_sentences()
        embeddings = load_embeddings()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

    index_type = st.selectbox(
        "Select FAISS Index Type",
        ("FlatL2", "IVFFlat", "IVFPQ"),
        format_func=lambda x: {
            "FlatL2": "Flat L2 (Exact)",
            "IVFFlat": "IVF Flat (Partitioned)",
            "IVFPQ": "IVF PQ (Partitioned + Quantized)"
        }[x]
    )

    st.write(f"Building {index_type} index...")
    try:
        index = build_faiss_index(embeddings, index_type)
    except Exception as e:
        st.error(f"An error occurred while building the FAISS index: {e}")
        st.stop()

    query = st.text_input("Enter your query sentence:", "something that is related to computer science and software engineering")
    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a valid query sentence.")
        else:
            try:
                results, elapsed = search(index, model, sentences, query, k=5, device="cpu")
                st.write(f"Search took {elapsed:.4f} seconds.")
                st.markdown("### Top 5 Results:")
                for i, (sent, dist) in enumerate(results, 1):
                    st.write(f"**{i}.** {sent}  \n_L2 Distance: {dist:.4f}_")
            except Exception as e:
                st.error(f"An error occurred during the search: {e}")

if __name__ == "__main__":
    main()
