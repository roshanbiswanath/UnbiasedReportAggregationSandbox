import random
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve
import chromadb
from collections import defaultdict

# Load the Multi-News dataset (train split)
dataset = load_dataset("multi_news", split="train")

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract articles and assign cluster IDs
articles = []
cluster_ids = []
for idx, example in enumerate(dataset):
    docs = example["document"].split(" ||||| ")
    for doc in docs:
        articles.append(doc.strip())
        cluster_ids.append(idx)

# Generate embeddings for all articles
embeddings = model.encode(articles, show_progress_bar=True)

# Group articles by cluster
cluster_to_articles = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_to_articles[cluster_id].append(idx)

# Sample same-cluster pairs
same_cluster_pairs = []
for cluster, idxs in cluster_to_articles.items():
    if len(idxs) >= 2:
        pairs = [(i, j) for i in idxs for j in idxs if i < j]
        same_cluster_pairs.extend(pairs)

# Limit to 10,000 same-cluster pairs
if len(same_cluster_pairs) > 10000:
    same_cluster_pairs = random.sample(same_cluster_pairs, 10000)

# Sample different-cluster pairs
different_cluster_pairs = []
while len(different_cluster_pairs) < 10000:
    i, j = random.sample(range(len(articles)), 2)
    if cluster_ids[i] != cluster_ids[j]:
        different_cluster_pairs.append((i, j))

# Function to calculate cosine distances between pairs
def calculate_distances(pairs, embeddings):
    distances = []
    for i, j in pairs:
        emb_i = embeddings[i]
        emb_j = embeddings[j]
        distance = cosine(emb_i, emb_j)
        distances.append(distance)
    return distances

# Calculate distances
same_distances = calculate_distances(same_cluster_pairs, embeddings)
different_distances = calculate_distances(different_cluster_pairs, embeddings)

# Prepare data for ROC curve analysis
all_distances = np.array(same_distances + different_distances)
labels = np.array([1] * len(same_distances) + [0] * len(different_distances))

# Compute ROC curve and find optimal threshold
fpr, tpr, thresholds = roc_curve(labels, -all_distances)  # Negative distances since smaller means more similar
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold determined: {optimal_threshold}")

# Set up ChromaDB with cosine distance metric
client = chromadb.Client()
collection = client.create_collection("news_articles")

# Add embeddings to ChromaDB
for i, emb in enumerate(embeddings):
    collection.add(ids=[str(i)], embeddings=[emb.tolist()], metadatas=[{"cluster_id": cluster_ids[i]}])

# Example: Query with a new article
new_article = "Sample text of a new news article to test similarity."
new_embedding = model.encode([new_article])[0]

# Query ChromaDB for top-10 nearest neighbors
results = collection.query(query_embeddings=[new_embedding.tolist()], n_results=10)
distances = results['distances'][0]
ids = results['ids'][0]

# Filter neighbors within the optimal threshold
similar_ids = [ids[i] for i in range(len(distances)) if distances[i] < optimal_threshold]

if similar_ids:
    print("Articles covering the same event found with IDs:", similar_ids)
else:
    print("No articles found covering the same event.")


    