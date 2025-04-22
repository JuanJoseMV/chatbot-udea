import argparse
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from uuid import uuid4

def main(model_name, data_path, db_path):
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"uri": db_path},
    )

    # Load data
    with open(data_path, 'r') as f:
        articles = json.load(f)

    # Add items to vector store
    documents = [Document(page_content=data["texto"], metadata={"articulo": article.split(" ")[1]}) for article, data in articles.items()]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    print(f"Adding {len(documents)} documents to the vector store...")
    try:
        vector_store.add_documents(documents=documents, ids=uuids)
        print("Documents added successfully.")
    except Exception as e:
        print(f"An error occurred while adding documents to the vector store: \n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to save embeddings")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the database')

    args = parser.parse_args()
    main(args.model_name, args.data_path, args.db_path)