import json
import random

topics = [
    "Machine Learning",
    "Deep Learning",
    "Neural Networks",
    "Blockchain",
    "Web3",
    "Decentralized Finance",
    "Smart Contracts",
    "Ethereum",
    "NFTs",
    "Metaverse",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Data Mining",
    "Big Data",
    "Quantum Computing",
    "Edge AI",
    "Federated Learning",
    "Cloud Computing",
]

templates = [
    "{} is a revolutionary field that is changing how we approach {} in the modern world.",
    "Applications of {} include {} across industries like healthcare, finance, and education.",
    "The future of {} depends on advancements in {} and global adoption.",
    "{} plays a crucial role in enabling {} through scalability and security.",
    "{} integrates with {} to provide innovative solutions for next-generation systems.",
    "Researchers are focusing on how {} can solve challenges related to {}.",
]

documents = []
for i in range(1, 5001):  # 5000 documents
    t1, t2 = random.sample(topics, 2)
    text = random.choice(templates).format(t1, t2)
    documents.append(text)

query = "How Web3 and AI can work together?"

data = {"query": query, "documents": documents}

with open("large_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ large_dataset.json generated with", len(documents), "documents")
