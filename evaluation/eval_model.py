import json
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
import argparse
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)


def main(model_name: str, articles_path: str, save_folder: str) -> None:
    """
    Evaluate a model with different dimensions.

    Args:
    -----
    - model_name: str
        SentenceTransformer model ID.
    - articles_path: str
        Path to the articles JSON file.
    - save_folder: str
        Path to save the evaluation results.

    Returns:
    --------
    - None
    
    """
    # Define the dimensions to evaluate
    matryoshka_dimensions = [768, 512, 256, 128, 64] # IMPORTANT: from the largest to the smallest

    # Load a model
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(
        model_name, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the dataset from a JSON file
    print("Loading dataset...")
    with open(articles_path, 'r') as f:
        data = json.load(f)

    # Extract the text from the articles and the questions
    print("Extracting data...")
    corpus = {} # (cid, text)
    queries = {} # (qid, text)
    relevant_docs = {} # (qid, set([relevant_cids]))

    for article, article_data in data.items():
        cid = article
        corpus[cid] = article_data["texto"]
        questions = article_data["preguntas"]
        questions = list(questions.keys())

        for i, question in enumerate(questions):
            qid = f"{cid}_{i}"
            queries[qid] = question
            previous_qid = relevant_docs.get(qid, set())
            relevant_docs[qid] = previous_qid.union([cid])

    matryoshka_evaluators = []
    # Iterate over the different dimensions
    for dim in tqdm(matryoshka_dimensions, desc="Creating evaluators"):
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
            show_progress_bar=True,
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create a sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)

    # Evaluate the model
    results = evaluator(model)

    # Save all the results (dict) to json file
    save_path = f"{save_folder}/{model_name}_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f)

    print(f"Results for model {model_name}")
    print(f"Evaluator results: \n{results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with different dimensions.")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="Hugging Face model ID")
    parser.add_argument("--articles_path", type=str, default="data/articles_expansion.json", help="Path to the articles JSON file")
    parser.add_argument("--save_folder", type=str, default="evaluation/results/", help="Path to save the evaluation results")

    args = parser.parse_args()
    main(args.model_name, args.articles_path, args.save_folder)