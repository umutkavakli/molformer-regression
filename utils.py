import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

from peft.peft import PEFT
from data import SMILESDataset
from model import MoLFormerWithRegressionHead

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

def plot_model_loss(stats, save_name):
    """
    Plot training and validation loss curves and save the figure.
    
    Args:
        stats: Dictionary containing training statistics with loss histories
        save_name: Base name for the saved plot file
    """
    plt.plot(range(1, len(stats["losses"]["train"]) + 1), stats["losses"]["train"], label="Training Loss")
    plt.plot(range(1, len(stats["losses"]["val"]) + 1), stats["losses"]["val"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{save_name}_loss_plot.png")


def get_dataset(dataset, tokenizer, test_size=0.2, external_data=None, dataset_type="original", return_longest_tokenized_sequence=False):
    """
    Prepare and split datasets for training, validation, and testing.
    
    This function:
    1. Converts the dataset to pandas
    2. Splits data into train/val/test sets
    3. Analyzes sequence lengths
    4. Creates SMILESDataset objects
    
    Args:
        dataset: Input dataset containing SMILES strings and labels
        tokenizer: Tokenizer for processing SMILES strings
        test_size: Proportion of data to use for testing (default: 0.2)
        external_data: Optional external data to incorporate
        dataset_type: Type of dataset to create - "original", "external", or "both"
        return_longest_tokenized_sequence: Whether to return max sequence length
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, [max_sequence_length])
    """
    # shuffle dataset
    total_data = dataset['train'].to_pandas()

    # split the data into training and test datasets - (0.8 train, 0.2 test)
    train_split, test_split = train_test_split(total_data, test_size=test_size, random_state=42) 

    # split test data half to have validation set (0.1 val, 0.1 test in original)
    val_split, test_split = train_test_split(test_split, test_size=0.5, random_state=42)

    # calculate longest input string and tokenized input string
    longest_sequence = max(len(sequence) for sequence in total_data['SMILES'])
    longest_tokenized_sequence = max(len(tokenizer(total_data["SMILES"][i])["input_ids"]) for i in range(len(total_data)))

    print(f"Column names in the dataset: {total_data.columns}")
    print(f"Longest input sequence in SMILES dataset: {longest_sequence}")
    print(f"Longest tokenized input sequence in SMILES dataset: {longest_tokenized_sequence}\n")
    print(f"Random 3 data sample out of {len(total_data)}:")
    for i in range(3):
        idx = random.randint(0, len(total_data))
        smiles, label = total_data['SMILES'][idx], total_data['label'][idx]
        print(f"\tSample {i+1}, [INPUT STRING]: {smiles} | [TARGET]: {label}")

    # print dataset size
    print(
        f"\nTraining Set Size: {len(train_split)}\n"
        f"Validation Set Size: {len(val_split)}\n"
        f"Test Set Size: {len(test_split)}"
    )

    # creating train, validation and test datasets
    train_dataset = SMILESDataset(train_split, tokenizer, max_length=longest_tokenized_sequence, external_data=external_data, dataset_type=dataset_type)
    val_dataset = SMILESDataset(val_split, tokenizer, max_length=longest_tokenized_sequence)
    test_dataset = SMILESDataset(test_split, tokenizer, max_length=longest_tokenized_sequence)

    if return_longest_tokenized_sequence:
        return train_dataset, val_dataset, test_dataset, longest_tokenized_sequence
    else:
        return train_dataset, val_dataset, test_dataset
    
def get_trainers(mode, data_selection, model_type, batch_size, learning_rate, external_data_path, weights, freeze_layers, peft_type, rank):
    # load pre-trained tokenizer from Huggingface
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # load dataset from huggingface
    dataset = load_dataset(DATASET_PATH)
    
    # get dataset
    if external_data_path:
        external_df = pd.read_csv(f"{external_data_path}")
        train_dataset, val_dataset, test_dataset, longest_sequence = get_dataset(dataset, tokenizer, external_data=external_df, dataset_type="both", return_longest_tokenized_sequence=True)
        external_dataset = SMILESDataset(external_df, tokenizer, max_length=longest_sequence, dataset_type="external")
    else:
        train_dataset, val_dataset, test_dataset = get_dataset(dataset, tokenizer)

    # choosing device, cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creating dataloaders, if it is mlm, then it needs random masking for inputs
    if model_type == "mlm":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if mode is None:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if data_selection == "influence":
            model = MoLFormerWithRegressionHead(model, is_mlm=True)
            model.load_state_dict(torch.load(f"{weights}", weights_only=True))
            print("\n[INFO]: Influence-based data selection method will be used.")
        elif data_selection == "similarity":
            model.load_state_dict(torch.load(f"{weights}", weights_only=True))
            print("\n[INFO]: Similarity-based data selection method will be used.")

    elif mode == "train":
        if model_type == "regression":
            model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
            model = MoLFormerWithRegressionHead(model, is_mlm=False, freeze=freeze_layers)
            print("\n[INFO]: Regression model will be used without MLM finetuning for training.")
        elif model_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            print("\n[INFO]: MLM model will be used for training.")  
        elif model_type == "mlm_regression":
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model.load_state_dict(torch.load(weights, weights_only=True))
            model = MoLFormerWithRegressionHead(model, is_mlm=True, freeze=freeze_layers)
            print("\n[INFO]: Regression model will be used with MLM finetuning for training.")
    

    elif mode == "test":
        if model_type == "regression":
            model = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
            model = MoLFormerWithRegressionHead(model, is_mlm=False, freeze=freeze_layers)
            print("\n[INFO]: Regression model will be used without MLM finetuning for testing.")
        elif model_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            print("\n[INFO]: MLM model will be used for testing.")  
        elif model_type == "mlm_regression":
            model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model = MoLFormerWithRegressionHead(model, is_mlm=True, freeze=freeze_layers)
            print("\n[INFO]: Regression model will be used with MLM finetuning for testing.")
        
        model.load_state_dict(torch.load(weights, weights_only=True))

    
    if peft_type:
        peft = PEFT(model)
        
        if peft_type == "bitfit":
            peft.apply_bitfit()
        elif peft_type == "lora":
            peft.apply_lora(rank=rank, alpha=rank*2)
        elif peft_type == "ia3":
            peft.apply_ia3()

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
    scheduler = StepLR(optimizer, 10)

    return (
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_dataset,
        val_dataset,
        test_dataset,
        external_df if external_data_path else None,
        external_dataset if external_data_path else None,
        criterion,
        optimizer,
        scheduler,
        device
    )

    
