import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    """
    A PyTorch Dataset for handling SMILES string data for molecular property prediction.
    
    This dataset class tokenizes SMILES strings and prepares them for model training,
    supporting multiple dataset configurations including original data, external data,
    or a combination of both.
    """
    def __init__(self, data, tokenizer, max_length=512, external_data=None, dataset_type="original"):
        """
        Initialize the SMILES dataset.
        
        Args:
            data: DataFrame containing SMILES strings and their associated labels
            tokenizer: Tokenizer used to convert SMILES strings to token IDs
            max_length: Maximum sequence length for tokenization (default: 512)
            external_data: Optional external DataFrame with additional SMILES data
            dataset_type: Type of dataset to create - "original", "external", or "both"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        if dataset_type == "original":
            self.smiles = data["SMILES"].tolist()
            self.labels = data["label"].tolist()
        elif dataset_type == "external":
            self.smiles = data["SMILES"].tolist()
            self.labels = data["Label"].tolist()
        elif dataset_type == "both":
            self.smiles = data["SMILES"].tolist() + external_data["SMILES"].tolist()
            self.labels = data["label"].tolist() + external_data["Label"].tolist()
        else:
            print("Unknown dataset type. Options: ['original', 'external', 'both']")

        
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Dataset size
        """
        return len(self.smiles)
    
    def __getitem__(self, idx):
        """
        Retrieve and process a sample from the dataset at the given index.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels for the model
        """
        smiles = self.smiles[idx]
        label = self.labels[idx]

        # tokenize input using model's tokenizer
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "input_ids": encoding['input_ids'].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32)
        }