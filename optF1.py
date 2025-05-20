import chromadb
import datasets
from chromadb.utils import embedding_functions
from concurrent.futures import ThreadPoolExecutor
import progressbar
import random
from itertools import groupby
import numpy as np
import math
import os
import shutil

# --- Configuration ---
MODEL_NAME = "all-MiniLM-L6-v2" # Model name for SentenceTransformer
# Use "cuda" if you have a compatible NVIDIA GPU and PyTorch/TF installed with CUDA support
# Use "cpu" otherwise
DEVICE = "cuda"
NORMALIZE_EMBEDDINGS = True
DB_BASE_PATH = "./multi_news_vectors" # Base directory for DBs
BATCH_SIZE_QUERY = 500  # Batch size for querying NNs
BATCH_SIZE_ADD = 256  # Batch size for adding to DB
# For Ryzen 9 (12 cores/24 threads), 16-20 workers is often a good starting point
MAX_WORKERS = 16
# Set to True to delete existing DB directories and recreate collections
# Set to False to try and use existing collections (will rebuild mappings)
RECREATE_DB = False
# --- End Configuration ---

# --- Helper Functions ---

def setup_progressbar(description, max_value):
    """Creates and starts a progress bar."""
    widgets = [
        f'{description}: ', progressbar.Percentage(), ' ',
        progressbar.Bar(marker='#', left='[', right=']'), ' ',
        progressbar.ETA(), ' ', progressbar.SimpleProgress()
    ]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=max_value).start()
    return bar

def prepare_mappings_and_ids(dataset_split, split_name_str): # <-- NEW
    """Creates docId->cluster and docId->text mappings and lists for ingestion."""
    # print(f"Preparing mappings for {dataset_split.dataset_name} split...") # <-- OLD
    print(f"Preparing mappings for {split_name_str} split...") # <-- NEW
    all_docs = []
    all_metadatas = []
    all_ids = []
    docId_to_cluster = {}
    docId_to_text = {}
    current_id_counter = 0

    for idx, example in enumerate(dataset_split):
        docs = example["document"].split(" ||||| ")
        for doc in docs:
            doc_text = doc.strip()
            if not doc_text: continue # Skip empty documents

            docId = str(current_id_counter)
            docId_to_cluster[docId] = idx # Store cluster index (relative to this split)
            docId_to_text[docId] = doc_text

            all_docs.append(doc_text)
            all_metadatas.append({"cluster_id": idx}) # Store cluster ID if needed later
            all_ids.append(docId)
            current_id_counter += 1

    print(f"Prepared {len(all_ids)} documents.")
    return docId_to_cluster, docId_to_text, all_docs, all_metadatas, all_ids

def get_or_create_collection(client, collection_name, embedding_function):
    """Gets an existing collection or creates a new one."""
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        print(f"Connected to existing collection '{collection_name}'.")
        return collection, False # Return False indicating it wasn't created now
    except Exception:
        print(f"Collection '{collection_name}' not found. Creating...")
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "l2"} # Use l2 for Euclidean distance
        )
        print(f"Collection '{collection_name}' created.")
        return collection, True # Return True indicating it was just created

def populate_collection(collection, all_ids, all_docs, all_metadatas, batch_size_add):
    """Adds documents to the collection if it's empty or newly created."""
    print(f"Adding {len(all_ids)} documents to collection '{collection.name}'...")
    total_docs = len(all_ids)
    bar_add = setup_progressbar(f'Ingesting {collection.name}', total_docs)

    for i in range(0, total_docs, batch_size_add):
        batch_ids = all_ids[i:i + batch_size_add]
        batch_docs = all_docs[i:i + batch_size_add]
        batch_metadatas = all_metadatas[i:i + batch_size_add]

        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metadatas
            )
        except Exception as e:
             print(f"\nError adding batch to {collection.name} starting at index {i}: {e}")
        bar_add.update(min(i + batch_size_add, total_docs))

    bar_add.finish()
    print(f"Finished adding documents to collection '{collection.name}'.")


# Function to process a single batch for NN distance calculation
def process_nn_batch(batch_data):
    batch, collection, docId_to_cluster = batch_data # Unpack data
    texts = [text for _, text in batch]
    docIds = [docId for docId, _ in batch]
    batch_d_same = []
    batch_d_diff = []

    try:
        results = collection.query(
            query_texts=texts,
            n_results=2, # Find self + 1 NN
            include=["distances"]
        )

        for i in range(len(texts)):
            if not results or not results["ids"] or len(results["ids"]) <= i or not results["ids"][i]:
                continue

            distances = results["distances"][i]
            ids = results["ids"][i]

            nn_distance = None
            nn_id = None
            if ids[0] != docIds[i]:
                nn_distance = distances[0]
                nn_id = ids[0]
            elif len(ids) > 1:
                nn_distance = distances[1]
                nn_id = ids[1]
            else:
                continue # Only found itself

            if nn_id is None or nn_id not in docId_to_cluster:
                 continue # Skip if NN not in our mapping for this split

            nn_cluster = docId_to_cluster[nn_id]
            true_cluster = docId_to_cluster[docIds[i]]

            if nn_cluster == true_cluster:
                batch_d_same.append(nn_distance)
            else:
                batch_d_diff.append(nn_distance)

    except Exception as e:
        print(f"\nError processing NN batch in {collection.name}: {e}")

    return batch_d_same, batch_d_diff


def calculate_nn_distances(collection, docId_to_cluster, docId_to_text, batch_size_query, max_workers):
    """Calculates same and different cluster NN distances for a given collection."""
    print(f"Calculating NN distances for collection '{collection.name}'...")
    query_items = list(docId_to_text.items())
    query_batches = [query_items[i:i + batch_size_query] for i in range(0, len(query_items), batch_size_query)]

    d_same = []
    d_diff = []

    bar_query = setup_progressbar(f'Querying {collection.name}', len(query_batches))

    # Prepare arguments for mapping - each item needs the batch, collection, and mapping
    map_args = [(batch, collection, docId_to_cluster) for batch in query_batches]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_results = executor.map(process_nn_batch, map_args)
        for i, (batch_d_same, batch_d_diff) in enumerate(future_results):
            d_same.extend(batch_d_same)
            d_diff.extend(batch_d_diff)
            bar_query.update(i + 1)

    bar_query.finish()
    print(f"Collected {len(d_same)} same-cluster and {len(d_diff)} different-cluster NN distances for {collection.name}.")
    return d_same, d_diff


def find_optimal_threshold_f1(d_same, d_diff):
    """Finds the threshold maximizing F1 score."""
    print("Calculating optimal threshold based on F1 score...")
    distances = [(d, 1) for d in d_same] + [(d, 0) for d in d_diff]
    if not distances:
        print("Warning: No distances found to calculate threshold.")
        return 0.0, 0.0, {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "Precision": 0, "Recall": 0}

    distances.sort(key=lambda x: x[0])

    total_d_same = len(d_same) # Actual Positives
    total_d_diff = len(d_diff) # Actual Negatives

    s_d_same = 0 # Count of same_cluster distances processed (TP below threshold)
    s_d_diff = 0 # Count of diff_cluster distances processed (FP below threshold)

    best_f1 = -1.0
    best_t = distances[0][0]
    best_metrics = {"TP": 0, "FP": 0, "FN": total_d_same, "TN": total_d_diff, "Precision": 0, "Recall": 0}

    # Iterate through sorted distances
    for d, group in groupby(distances, key=lambda x: x[0]):
        # --- Calculate metrics for threshold `d` ---
        tp = s_d_same
        fp = s_d_diff
        fn = total_d_same - tp
        tn = total_d_diff - fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_d_same if total_d_same > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 >= best_f1: # Prefer higher F1, or lower threshold if F1 is tied
             if f1 > best_f1 or d < best_t: # Update if better F1 or same F1 at lower distance
                best_f1 = f1
                best_t = d
                best_metrics = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": precision, "Recall": recall}

        # --- Update counts for the next iteration ---
        for _, label in group:
            if label == 1: s_d_same += 1
            else: s_d_diff += 1

    # --- Check the scenario AFTER the last distance (classify all as same) ---
    tp = total_d_same
    fp = total_d_diff
    fn = 0
    tn = 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_d_same if total_d_same > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if f1 > best_f1:
        best_f1 = f1
        best_t = distances[-1][0] + 1e-6 # Threshold slightly above max distance
        best_metrics = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": precision, "Recall": recall}

    print(f"Optimal threshold found: {best_t:.6f} with F1: {best_f1:.4f}")
    return best_t, best_f1, best_metrics

def evaluate_threshold(d_same, d_diff, threshold):
    """Evaluates a given threshold on a set of distances."""
    print(f"Evaluating threshold {threshold:.6f}...")
    tp = sum(1 for d in d_same if d < threshold)
    fn = len(d_same) - tp
    fp = sum(1 for d in d_diff if d < threshold)
    tn = len(d_diff) - fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # tp+fn is len(d_same)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": precision, "Recall": recall, "F1": f1}
    print(f"Evaluation Results: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    return metrics

# --- Main Execution ---

# 1. Initialize Embedding Function and ChromaDB Client
print("Initializing embedding function...")
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME, device=DEVICE, normalize_embeddings=NORMALIZE_EMBEDDINGS
)

# Handle DB Recreation
if RECREATE_DB and os.path.exists(DB_BASE_PATH):
    print(f"Recreating DB: Deleting existing directory '{DB_BASE_PATH}'...")
    shutil.rmtree(DB_BASE_PATH)
if not os.path.exists(DB_BASE_PATH):
     os.makedirs(DB_BASE_PATH)

client = chromadb.PersistentClient(path=DB_BASE_PATH)


# --- Process Splits ---
results = {}
optimal_train_threshold = 0.0

for split_name in ["train", "validation", "test"]:
    print(f"\n===== Processing {split_name.upper()} Split =====")

    # Load Dataset Split
    print(f"Loading Multi-News '{split_name}' split...")
    try:
        dataset = datasets.load_dataset("multi_news", split=split_name)
    except Exception as e:
        print(f"Error loading dataset split '{split_name}': {e}")
        continue # Skip to next split if loading fails

    # Prepare Mappings
    docId_to_cluster, docId_to_text, all_docs, all_metadatas, all_ids = prepare_mappings_and_ids(dataset,split_name)

    if not all_ids:
        print(f"No documents found for split '{split_name}'. Skipping.")
        continue

    # Get or Create Collection
    collection_name = f"multi_news_{split_name}"
    collection, created_now = get_or_create_collection(client, collection_name, emb_func)

    # Populate if needed (newly created or RECREATE_DB is True)
    # Check count to avoid repopulating if RECREATE_DB=False but collection exists
    needs_populating = created_now or RECREATE_DB
    if not needs_populating:
        try:
            count = collection.count()
            if count != len(all_ids):
                 print(f"Collection '{collection_name}' exists but count ({count}) differs from expected ({len(all_ids)}). Repopulating.")
                 # Consider deleting and recreating if counts mismatch significantly
                 client.delete_collection(name=collection_name)
                 collection = client.create_collection(name=collection_name, embedding_function=emb_func, metadata={"hnsw:space": "l2"})
                 needs_populating = True
            else:
                 print(f"Collection '{collection_name}' already populated with {count} items.")
        except Exception as e:
             print(f"Error checking collection count for {collection_name}, assuming population needed: {e}")
             needs_populating = True # Populate if unsure

    if needs_populating:
        populate_collection(collection, all_ids, all_docs, all_metadatas, BATCH_SIZE_ADD)


    # Calculate NN Distances
    d_same, d_diff = calculate_nn_distances(collection, docId_to_cluster, docId_to_text, BATCH_SIZE_QUERY, MAX_WORKERS)

    if not d_same and not d_diff:
        print(f"No NN distances calculated for {split_name}. Cannot proceed.")
        results[split_name] = {"error": "No distances calculated"}
        continue

    # --- Logic specific to split ---
    if split_name == "train":
        # Find optimal threshold on Train data
        optimal_train_threshold, train_f1, train_metrics = find_optimal_threshold_f1(d_same, d_diff)
        results[split_name] = {
            "optimal_threshold": optimal_train_threshold,
            "f1": train_f1,
            "metrics": train_metrics,
            "d_same_count": len(d_same),
            "d_diff_count": len(d_diff)
        }
        print(f"\n--- Train Set Results ---")
        print(f"Optimal Threshold Determined: {optimal_train_threshold:.6f}")
        print(f"Achieved F1 on Train Set: {train_f1:.4f}")
        print(f"Metrics: {train_metrics}")

    else: # Validation and Test splits
        if split_name == "validation":
             print(f"\n--- Validation Set Results (using Train Threshold {optimal_train_threshold:.6f}) ---")
        else: # test
             print(f"\n--- Test Set Results (using Train Threshold {optimal_train_threshold:.6f}) ---")

        # Evaluate using the threshold found on the TRAIN set
        eval_metrics = evaluate_threshold(d_same, d_diff, optimal_train_threshold)
        results[split_name] = {
            "evaluated_threshold": optimal_train_threshold,
            "f1": eval_metrics["F1"],
            "metrics": eval_metrics,
            "d_same_count": len(d_same),
            "d_diff_count": len(d_diff)
        }

print("\n===== FINAL SUMMARY =====")
if "train" in results and "optimal_threshold" in results["train"]:
    print(f"Optimal Threshold found on Train set: {results['train']['optimal_threshold']:.6f}")
    print(f"  Train F1: {results['train']['f1']:.4f} (TP={results['train']['metrics']['TP']}, FP={results['train']['metrics']['FP']}, FN={results['train']['metrics']['FN']}, TN={results['train']['metrics']['TN']})")
else:
    print("Training phase did not complete successfully.")

if "validation" in results and "f1" in results["validation"]:
     print(f"  Validation F1 (using Train threshold): {results['validation']['f1']:.4f} (TP={results['validation']['metrics']['TP']}, FP={results['validation']['metrics']['FP']}, FN={results['validation']['metrics']['FN']}, TN={results['validation']['metrics']['TN']})")
else:
    print("Validation phase did not complete or was skipped.")

if "test" in results and "f1" in results["test"]:
     print(f"  Test F1 (using Train threshold): {results['test']['f1']:.4f} (TP={results['test']['metrics']['TP']}, FP={results['test']['metrics']['FP']}, FN={results['test']['metrics']['FN']}, TN={results['test']['metrics']['TN']})")
else:
     print("Test phase did not complete or was skipped.")

print("=======================")