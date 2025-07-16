# Simple Similarity Search Application using FAISS and Sentence Transformers

## Overview

This project demonstrates a semantic similarity search application using [FAISS](https://github.com/facebookresearch/faiss) for fast vector search and [Sentence Transformers](https://www.sbert.net/) for generating sentence embeddings.

## Features

- Loads and deduplicates sentences from multiple datasets.
- Generates embeddings using a pre-trained Sentence Transformer model 'bert-base-nli-mean-tokens'.
- Stores embeddings in chunked `.npy` files for efficient memory usage.
- Supports three FAISS index types:
  - Flat L2 (Exact)
  - IVF Flat (Partitioned)
  - IVF PQ (Partitioned + Quantized)
- Streamlit web interface for interactive semantic search.

## Project Structure
|--
|-- `src/`                  # Source code for the application
|   |-- `main.py`           # Main application script
|   |-- `faiss_index.py`    # FAISS index management
|   |-- `sentence_transformer.py`  # Sentence Transformer model loading
|-- `workspace/`            # Workspace for data processing and notebooks        
|   |-- `Faiss_intro.ipynb` # Jupyter notebook for data preparation and embedding generation
|   |-- `sim_sentences/`    # Directory for storing sentence embeddings
|       |-- `embeddings_56.npy`  # Example embeddings file
|-- `requirements.txt`      # Python package dependencies
|-- `README.md`             # Project documentation 
