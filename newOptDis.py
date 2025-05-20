import os
import itertools
import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configuration
hf_dataset_name = 'multi_news'           # HuggingFace dataset identifier
hf_split = 'train'                       # dataset split to use
embedding_model_name = 'all-MiniLM-L6-v2'
chromadb_dir = './chroma_db'

# Load HuggingFace dataset
dataset = load_dataset(hf_dataset_name, split=hf_split)
# Expect columns: ['article_id', 'source', 'group_id', 'text']

# Initialize embedding model
embedder = SentenceTransformer(embedding_model_name)

# Initialize ChromaDB client and collection
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=chromadb_dir))
collection = client.get_or_create_collection(name="multi_news_articles")

# Prepare embeddings and metadata
ids = []
texts = []
metadatas = []
for example in dataset:
    ids.append(str(example['article_id']))
    texts.append(example['text'])
    metadatas.append({'source': example['source'], 'group_id': example['group_id']})

embeddings = embedder.encode(texts, show_progress_bar=True)

# Add to ChromaDB (overwrite existing if re-running)
if collection.count() > 0:
    collection.delete(where={})
collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
client.persist()

# Build pairwise distances and labels
def compute_euclidean(a, b):
    return np.linalg.norm(a - b)

# Map IDs to embeddings and metadata
emb_map = {i: emb for i, emb in zip(ids, embeddings)}
meta_map = {i: m for i, m in zip(ids, metadatas)}

pairs = []  # (distance, label)
for id1, id2 in itertools.combinations(ids, 2):
    m1, m2 = meta_map[id1], meta_map[id2]
    # only cross-source comparisons
    if m1['source'] == m2['source']:
        continue
    dist = compute_euclidean(emb_map[id1], emb_map[id2])
    same = (m1['group_id'] == m2['group_id'])
    pairs.append((dist, same))

# Split distances and labels
dists = np.array([p[0] for p in pairs])
labels = np.array([p[1] for p in pairs])

# Grid search for best threshold
grid = np.linspace(dists.min(), dists.max(), 100)
best_thresh, best_f1 = None, 0.0
for t in grid:
    preds = dists <= t
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"Optimal distance threshold: {best_thresh:.4f} with F1-score: {best_f1:.4f}")