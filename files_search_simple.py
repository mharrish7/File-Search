import os
import qdrant_client
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import argparse

model = SentenceTransformer('all-MiniLM-L6-v2')

# Qdrant configuration
qdrant_client = qdrant_client.QdrantClient(":memory:")  
collection_name = "file_embeddings"
vector_size = 384  

# Create Qdrant collection
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
)

def generate_embeddings(text):
    """Generates embeddings for a given text using Sentence Transformers."""
    return model.encode(text).tolist()

def index_files(directory):
    """Indexes files in a directory using Sentence Transformer embeddings and Qdrant."""
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    embedding = generate_embeddings(file_content)
                    print(filepath, embedding)
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            models.PointStruct(
                                id=hash(filepath),  # Unique ID for each file
                                vector=embedding,
                                payload={"filepath": filepath},
                            )
                        ],
                    )

            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='latin-1') as file:
                        file_content = file.read()
                        embedding = generate_embeddings(file_content)
                        qdrant_client.upsert(
                            collection_name=collection_name,
                            points=[
                                models.PointStruct(
                                    id=hash(filepath),
                                    vector=embedding,
                                    payload={"filepath": filepath},
                                )
                            ],
                        )

                except:
                    print(f"Skipping file {filepath} due to encoding issues.")

            except Exception as e:
                print(f"Error processing file {filepath}: {e}")

def search_files(query, top_k=5):
    """Searches for files based on a query using Sentence Transformer embeddings and Qdrant."""
    query_embedding = generate_embeddings(query)
    search_results = qdrant_client.search(
        collection_name=collection_name,
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

    directory_to_index = args.directory
    index_files(directory_to_index)

    search_query = args.query
    results = search_files(search_query, top_k=args.top_k)

    print(f"Search results for '{search_query}':")
    for result in results:
        print(f"  File: {result.payload['filepath']}, Score: {result.score}")

if __name__ == "__main__":
    main()