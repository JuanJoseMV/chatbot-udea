import argparse
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from tqdm import tqdm

def main(model_name, data_path, db_path, top_k):
    # Load model
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    # Load vector store
    vector_store = Milvus(
        embedding_function=embedder,
        connection_args={"uri": db_path},
    )

    # Load data
    with open(data_path, 'r') as f:
        articles = json.load(f)

    # Set retriever
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": top_k}
    )

    # Evaluate extraction
    scores = {}

    for article, data in tqdm(articles.items(), desc="Articles evaluated", total=len(articles)):
        queries = data["preguntas"]
        scores[article] = [] 

        for query, _ in queries.items():
            top_k_results = retriever.invoke(query)

            # TODO: replace technical words to layman words
            ...

            # search if article is in top k results' metadata's article
            found = False
            for i, result in enumerate(top_k_results):
                article_number = article.split(" ")[1]

                if result.metadata["articulo"] == article_number:
                    scores[article].append((query, i))
                    found = True
                    break

            if not found:
                scores[article].append((query, top_k + 1))

    # Save scores
    output_model_name = model_name.split("/")[-1]
    with open(f"evaluation/results/extraction_scores_{output_model_name}.json", 'w+', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to save embeddings")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--db_path', type=str, required=True, help='Path to the database')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to retrieve')

    args = parser.parse_args()
    main(args.model_name, args.data_path, args.db_path, args.top_k)