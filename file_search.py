import os
import qdrant_client
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import argparse
import hashlib
import json

MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = "file_embeddings"
VECTOR_SIZE = 384
DB_PATH = "qdrant_db"
INDEXED_FILES_PATH = "indexed_files.json"

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while chunk := file.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_indexed_files():
    """Loads the dictionary of already indexed files."""
    if os.path.exists(INDEXED_FILES_PATH):
        with open(INDEXED_FILES_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_indexed_files(indexed_files):
    """Saves the dictionary of indexed files."""
    with open(INDEXED_FILES_PATH, 'w') as f:
        json.dump(indexed_files, f)

def initialize_qdrant_client():
    """Initializes the Qdrant client."""
    return qdrant_client.QdrantClient(path=DB_PATH)

def initialize_collection(client):
    """Initializes the Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )

def generate_embeddings(text):
    """Generates embeddings for a given text using Sentence Transformers."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(text).tolist()

def index_files(directory, client, indexed_files):
    """Indexes files in a directory using Sentence Transformer embeddings and Qdrant."""
    for root, _, files in os.walk(directory):
        for filename in files:
            try:
                filepath = os.path.join(root, filename)
                filepath = os.path.abspath(filepath)
                file_hash = get_file_hash(filepath)

                if indexed_files.get(filepath) == file_hash:
                    continue

                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    embedding = generate_embeddings(file_content)
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            models.PointStruct(
                                id=hash(filepath),
                                vector=embedding,
                                payload={"filepath": filepath},
                            )
                        ],
                    )
                    indexed_files[filepath] = file_hash
                    print(f"Indexed: {filepath}")

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

    save_indexed_files(indexed_files)

def search_files(query, client, top_k=5):
    """Searches for files based on a query using Sentence Transformer embeddings and Qdrant."""
    query_embedding = generate_embeddings(query)
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    return search_results

def main():
    parser = argparse.ArgumentParser(description="Search files using Qdrant and Sentence Transformers.")
    parser.add_argument("query", help="The search query.")
    parser.add_argument("-d", "--directory", default="./", help="Directory to index.")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of search results to return.")
    args = parser.parse_args()

    client = initialize_qdrant_client()
    initialize_collection(client)
    indexed_files = load_indexed_files()
    index_directory = os.path.abspath(args.directory)
    index_files(index_directory, client, indexed_files)

    results = search_files(args.query, client, args.top_k)

    print(f"Search results for '{args.query}':")
    for result in results:
        print(f"  File: {result.payload['filepath']}, Score: {result.score}")

if __name__ == "__main__":
    main()