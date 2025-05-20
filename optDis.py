import chromadb
import datasets
from chromadb.utils import embedding_functions
import progressbar
import json

# Load the multi_news dataset (test split)
multi_news = datasets.load_dataset("multi_news", split="test")

# Initialize the embedding function
model_name = "all-MiniLM-L6-v2"
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, device="cuda", normalize_embeddings=True)

# Connect to the existing ChromaDB collection
client = chromadb.PersistentClient(path="./test_vectors")
collection = client.get_collection(name="multi_news", embedding_function=emb_func)

# Prepare mappings
docId_to_cluster = {}
docId_to_text = {}
for idx, example in enumerate(multi_news):
    docs = example["document"].split(" ||||| ")
    for i, doc in enumerate(docs):
        docId = str(len(docId_to_text))  # Sequential docIds starting from 0
        docId_to_cluster[docId] = idx
        docId_to_text[docId] = doc.strip()

# Collect distances to nearest neighbors with progress bar
d_same = []
d_diff = []

total_docs = len(docId_to_text)
widgets = [
    ' [',
    progressbar.Percentage(), ' ',
    '] ',
    progressbar.Bar('*'), ' (',
    progressbar.ETA(), ') ',
]
bar = progressbar.ProgressBar(maxval=total_docs, widgets=widgets)
bar.start()

for i, (docId, text) in enumerate(docId_to_text.items()):
    true_cluster = docId_to_cluster[docId]
    # Query the collection, excluding the document itself
    results = collection.query(query_texts=[text], n_results=2, where={"id": {"$ne": docId}})
    if len(results["distances"][0]) > 1 and results["ids"][0][1] != docId:
        nn_distance = results["distances"][0][1]
        nn_id = results["ids"][0][1]
    else:
        nn_distance = results["distances"][0][0]
        nn_id = results["ids"][0][0]
    nn_cluster = docId_to_cluster[nn_id]
    if nn_cluster == true_cluster:
        d_same.append(nn_distance)
    else:
        d_diff.append(nn_distance)
    bar.update(i)

bar.finish()

# Combine distances with labels
distances = [(d, 1) for d in d_same] + [(d, 0) for d in d_diff]
distances.sort(key=lambda x: x[0])

# Find optimal threshold
total_d_same = len(d_same)
total_d_diff = len(d_diff)
s_d_same = 0
s_d_diff = 0
min_error = float('inf')
best_t = None

from itertools import groupby
grouped = groupby(distances, key=lambda x: x[0])
for d, group in grouped:
    error = (total_d_same - s_d_same) + s_d_diff
    if error < min_error:
        min_error = error
        best_t = d
    for _, label in group:
        if label == 1:
            s_d_same += 1
        else:
            s_d_diff += 1

# Check for t > max distance
error = total_d_diff
if error < min_error:
    best_t = max([d[0] for d in distances]) + 1

print(f"Optimal distance threshold: {best_t}")
print(f"Same-cluster distances collected: {len(d_same)}")
print(f"Different-cluster distances collected: {len(d_diff)}")
print(f"Minimum error: {min_error}")