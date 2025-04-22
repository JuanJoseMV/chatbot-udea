"""
Code inspired from:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/matryoshka_nli.py
"""

import sys
import json
import logging
import argparse
import traceback
from datetime import datetime

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator, 
    SequentialEvaluator, 
    SimilarityFunction
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def main(
        model_name: str,
        dataset_path: str,
        train_config_file_path: str,
        save_output_dir: str,
) -> None:
    """
    Train a SentenceTransformer model with 🪆Matryoshka loss.

    Args:
    -----
    - model_name: str
        SentenceTransformer model ID.
    - dataset_path: str
        Path to the dataset.
    - train_config_file_path: str
        Path to the training configuration file.
    - save_output_dir: str
        Directory to save the output model.

    Returns:
    --------
    - None
    """
    batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
    num_train_epochs = 10
    matryoshka_dims = [768, 512, 256, 128, 64, 32]
    if "MiniLM" in model_name:
        matryoshka_dims = [384, 256, 128, 64, 32]

    output_dir = f"models/matryoshka_{model_name.split('/')[-1]}"

    # If not already a Sentence Transformer model, it will automatically
    # create one with "mean" pooling.
    model = SentenceTransformer(model_name)

    with open(dataset_path, 'r') as f:
        data = json.load(f)
        
    ## Extract the text from the articles and the questions
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

    # Split relevant_docs into train and eval datasets (80% train, 20% eval)
    # Create HuggingFace datasets
    qids = list(relevant_docs.keys())
    train_qids, eval_qids = train_test_split(qids, test_size=0.2, random_state=42)

    train_dict = {}
    eval_dict = {}

    train_dict["sentence_a"] = []
    train_dict["sentence_b"] = []
    train_dict["score"] = []

    eval_dict["sentence_a"] = []
    eval_dict["sentence_b"] = []
    eval_dict["score"] = []

    for qid in train_qids:
        for cid, text in corpus.items():
            train_dict["sentence_a"].append(queries[qid])
            train_dict["sentence_b"].append(text)
            train_dict["score"].append(1 if cid in relevant_docs[qid] else 0)

    for qid in eval_qids:
        for cid, text in corpus.items():
            eval_dict["sentence_a"].append(queries[qid])
            eval_dict["sentence_b"].append(text)
            eval_dict["score"].append(1 if cid in relevant_docs[qid] else 0)

    train_dataset = Dataset.from_dict(train_dict)
    eval_dataset = Dataset.from_dict(eval_dict)

    # Create train and eval datasets
    # train_dataset = Dataset.from_dict({
    #     "sentence_a": [queries[qid] for qid in train_qids],
    #     "sentence_b": [corpus[cid] for qid in train_qids for cid in relevant_docs[qid]],
    #     "score": [1.0] * len(train_qids),  # Dummy scores
    # })
    # eval_dataset = Dataset.from_dict({
    #     "sentence_a": [queries[qid] for qid in eval_qids],
    #     "sentence_b": [corpus[cid] for qid in eval_qids for cid in relevant_docs[qid]],
    #     "score": [1.0] * len(eval_qids),  # Dummy scores
    # })

    # TODO: try with different losses
    inner_train_loss = losses.CoSENTLoss(model)
    train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

    # Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
    matryoshka_evaluators = []
    ## Iterate over the different dimensions
    for dim in tqdm(matryoshka_dims, desc="Creating evaluators"):
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Important for Matryoshka loss
            score_functions={"cosine": cos_sim},
            show_progress_bar=True,
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    evaluator = SequentialEvaluator(matryoshka_evaluators)

    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=f"[UDA]matryoshka-{model_name.split('/')[-1]}",  # Will be used in W&B if `wandb` is installed
        report_to="wandb",  
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    final_output_dir = f"{save_output_dir}/trained_{model_name.split('/')[-1]}"
    model.save(final_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model with Matryoshka loss.")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="Name of the pre-trained model.")
    parser.add_argument("--dataset_path", type=str, default="data/articles_expansion.json", help="Path to the dataset.")
    parser.add_argument("--train_config_file_path", type=str, default="data/training_config.json", help="Path to the training configuration file.")
    parser.add_argument("--save_output_dir", type=str, default="models/", help="Directory to save the output model.")

    args = parser.parse_args()

    model_name = args.model_name
    dataset_path = args.dataset_path
    train_config_file_path = args.train_config_file_path
    save_output_dir = args.save_output_dir

    main(
        model_name=model_name,
        dataset_path=dataset_path,
        train_config_file_path=train_config_file_path,
        save_output_dir=save_output_dir,
    )