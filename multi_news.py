# Step 1: Import and load the dataset
from datasets import load_dataset
dataset = load_dataset("multi_news", trust_remote_code=True)

# Step 2: Define the extraction function
def extract_articles(split):
    articles = []
    for idx, example in enumerate(dataset[split]):
        docs = example["document"].split(" ||||| ")
        for doc in docs:
            articles.append({"text": doc.strip(), "cluster_id": idx})
    return articles

# Step 3: Extract articles from all splits
train_articles = extract_articles("train")
val_articles = extract_articles("validation")
test_articles = extract_articles("test")

# Step 4: Verify the first example
# print("First article:", train_articles[0]["text"][:200])  # First 200 characters
# print("Cluster ID:", train_articles[0]["cluster_id"])

# # Step 5: Example of accessing articles from the same cluster
# cluster_0_articles = [article["text"] for article in train_articles if article["cluster_id"] == 0]
# print("Number of articles in cluster 0:", len(cluster_0_articles))

for i in train_articles:
    print(i["cluster_id"])

# print(dataset["train"][1])