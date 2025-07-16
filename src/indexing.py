import faiss

def build_faiss_index(embeddings, index_type="FlatL2", nlist=50, m=8, bits=8):
    d = embeddings.shape[1]
    if index_type == "FlatL2":
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 5
    elif index_type == "IVFPQ":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10
    else:
        raise ValueError("Unknown index type")
    return index

def search(index, model, sentences, query, k=5, device="cpu"):
    import numpy as np
    import time
    xq = model.encode([query], device=device).astype(np.float32)
    start = time.time()
    D, I = index.search(xq, k)
    elapsed = time.time() - start
    results = [(sentences[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results, elapsed
