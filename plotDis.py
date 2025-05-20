import chromadb
import datasets
import json
import progressbar
from chromadb.utils import embedding_functions

multi_news = datasets.load_dataset("multi_news", split="test")
total_docs = len(multi_news)
print(f"Total documents in the dataset: {total_docs}")
model_name = "all-MiniLM-L6-v2"
# Initialize the embedding function
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name, device="cuda", normalize_embeddings=True)

# widgets = [' [',
# #TypeError: format requires a mapping
#             progressbar.Percentage(), ' ',
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]
 
# bar = progressbar.ProgressBar(maxval=total_docs, 
#                               widgets=widgets).start()



# Initialize the Chroma client
client = chromadb.PersistentClient(path="./test_vectors")
# Create a collection
# client.delete_collection(name="multi_news")
collection = client.get_or_create_collection(name="multi_news", embedding_function=emb_func)
# # Add documents to the collection
# docId = 0
# clusterToCount = {}
# bar.start()
# for idx, example in enumerate(multi_news):
#     docs = example["document"].split(" ||||| ")
#     clusterToCount[idx] = len(docs)
#     for doc in docs:
#         collection.add(documents=[doc.strip()],ids=[str(docId)] , metadatas=[{"cluster_id": idx}])
#         docId += 1
#         # print(f"Added document {docId} to the collection with cluster ID {idx}")
#     bar.update(idx)
# bar.finish()
# Save the cluster count to a JSON file

# f = open("multi_news_cluster_count.json", "w")
# f.write(json.dumps(clusterToCount))
# f.close()
# print("Cluster count saved to multi_news_cluster_count.json")

print("Documents added to the collection.")
# Query the collection
query_result = collection.query(
    query_texts=["This is a sample query."],
    n_results=5,
)

print("Query Results:")
for i, doc in enumerate(query_result['documents'][0]):
    # print(f"Document {i+1}: {doc}")
    print(f"Metadata: {query_result['metadatas'][0][i]}")
    print(f"Distance: {query_result['distances'][0][i]}")