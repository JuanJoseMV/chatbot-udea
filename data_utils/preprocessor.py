# from ...preprocessor import Preprocessor  
from datasets import DatasetDict

def load_dataset(ds_path, tokenizer, interactive, train_size=0.8):
    # Load dataset
    ...

    # Split dataset
    ...

    # Initialize preprocessor
    # preprocessor = Preprocessor(tokenizer=tokenizer, interactive=interactive)

    # Preprocess train and validation datasets
    ...

    # Concat train and validation datasets
    ...

    # Shuffle splits
    ...
    
def load_test_dataset(ds_path, tokenizer, interactive):
    ...