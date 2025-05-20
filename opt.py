import chromadb
import datasets
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor
import progressbar
import random

# Load the dataset (test split)
multi_news = datasets.load_dataset("multi_news", split="test")

# Initialize embedding function
model_name = "all-MiniLM-L6-v2"
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=model_name, device="cuda", normalize_embeddings=True
)

# Connect to ChromaDB collection
client = chromadb.PersistentClient(path="./test_vectors")
collection = client.get_collection(name="multi_news", embedding_function=emb_func)

# Prepare document mappings
docId_to_cluster = {}
docId_to_text = {}
for idx, example in enumerate(multi_news):
    docs = example["document"].split(" ||||| ")
    for doc in docs:
        docId = str(len(docId_to_text))  # Sequential IDs
        docId_to_cluster[docId] = idx
        docId_to_text[docId] = doc.strip()

# Sample a subset of documents
# sample_size = 1000  # Adjust as needed
# sampled_items = random.sample(list(docId_to_text.items()), min(sample_size, len(docId_to_text)))
sampled_items = list(docId_to_text.items())


# Create batches
batch_size = 50  # Adjust based on memory
batches = [sampled_items[i:i + batch_size] for i in range(0, len(sampled_items), batch_size)]

# Function to process a single batch
def process_batch(batch):
    texts = [text for _, text in batch]
    docIds = [docId for docId, _ in batch]
    results = collection.query(query_texts=texts, n_results=2, include=["distances"])
    batch_d_same = []
    batch_d_diff = []
    for i in range(len(texts)):
        distances = results["distances"][i]
        ids = results["ids"][i]
        if ids[0] != docIds[i]:
            nn_distance = distances[0]
            nn_id = ids[0]
        elif len(ids) > 1:
            nn_distance = distances[1]
            nn_id = ids[1]
        else:
            continue
        nn_cluster = docId_to_cluster[nn_id]
        true_cluster = docId_to_cluster[docIds[i]]
        if nn_cluster == true_cluster:
            batch_d_same.append(nn_distance)
        else:
            batch_d_diff.append(nn_distance)
    return batch_d_same, batch_d_diff

# Collect distances with parallel processing and progress bar
d_same = []
d_diff = []

# Initialize progress bar
widgets = [' [', progressbar.Percentage(), ' ', '] ', progressbar.Bar('*'), ' (', progressbar.ETA(), ') ']
bar = progressbar.ProgressBar(maxval=len(batches), widgets=widgets)
bar.start()

# Process batches in parallel
with ThreadPoolExecutor(max_workers=16) as executor:
    for batch_idx, (batch_d_same, batch_d_diff) in enumerate(executor.map(process_batch, batches)):
        d_same.extend(batch_d_same)
        d_diff.extend(batch_d_diff)
        bar.update(batch_idx)

bar.finish()

# Compute optimal threshold
distances = [(d, 1) for d in d_same] + [(d, 0) for d in d_diff]
distances.sort(key=lambda x: x[0])

total_d_same = len(d_same)
total_d_diff = len(d_diff)
s_d_same = 0
s_d_diff = 0
min_error = float('inf')
best_t = None

from itertools import groupby
for d, group in groupby(distances, key=lambda x: x[0]):
    error = (total_d_same - s_d_same) + s_d_diff
    if error < min_error:
        min_error = error
        best_t = d
    for _, label in group:
        if label == 1:
            s_d_same += 1
        else:
            s_d_diff += 1

# Check beyond max distance
if total_d_diff < min_error:
    best_t = max([d[0] for d in distances]) + 1

print(f"Optimal distance threshold: {best_t}")
print(f"Same-cluster distances: {len(d_same)}")
print(f"Different-cluster distances: {len(d_diff)}")
print(f"Minimum error: {min_error}")