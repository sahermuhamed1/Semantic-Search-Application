import os
import glob
import numpy as np

SENTENCES_PATH = "../workspace/sentences.txt"
EMBEDDINGS_PATH = "../workspace/sim_sentences/"  # Directory containing embedding chunks

def load_sentences():
    if not os.path.exists(SENTENCES_PATH):
        raise FileNotFoundError(f"Sentences file not found at {SENTENCES_PATH}")
    with open(SENTENCES_PATH, "r") as f:
        sentences = [line.strip() for line in f.readlines()]
    return sentences

def load_embeddings():
    embeddings_dir = EMBEDDINGS_PATH
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"Embeddings directory not found at {embeddings_dir}")
    def numeric_key(x):
        base = os.path.splitext(os.path.basename(x))[0]
        suffix = base.split("_")[1]
        return int(suffix) if suffix.isdigit() else float('inf')
    embeddings_files = sorted(
        glob.glob(os.path.join(embeddings_dir, "embeddings_*.npy")),
        key=numeric_key
    )
    embeddings_files = [f for f in embeddings_files if os.path.splitext(os.path.basename(f))[0].split("_")[1].isdigit()]
    if not embeddings_files:
        raise FileNotFoundError(f"No valid embeddings files found in {embeddings_dir}")
    embeddings_list = []
    for file in embeddings_files:
        arr = np.load(file)
        embeddings_list.append(arr)
    merged_embeddings = np.vstack(embeddings_list).astype(np.float32)
    return merged_embeddings
