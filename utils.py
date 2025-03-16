import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data import SMILESDataset

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