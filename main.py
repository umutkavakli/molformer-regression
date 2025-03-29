import argparse

DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_RANK = 64

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for training and evaluating molecular models with fine-tuning options."
    )

    parser.add_argument(
        "-m", "--mode", choices=["train", "test", "influence", "similarity"], default="train",
        help="Mode of operation: 'train', 'test', 'influence' or 'similarity'. Default: 'train'."
    )

    parser.add_argument(
        "-t", "--model-type", choices=["regression", "mlm", "mlm_regression"], default="mlm_regression",
        help="Model type: 'regression', 'mlm', or 'mlm_regression'. Default: 'mlm_regression'."
    )

    parser.add_argument(
        "-b", "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch Size. Default: {DEFAULT_BATCH_SIZE}."
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs. Default: {DEFAULT_EPOCHS}."
    )

    parser.add_argument(
        "-l", "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}."
    )

    parser.add_argument(
        "-s", "--save-weights-type", choices=["best", "last"], default="best",
        help="Weight saving strategy: 'best' or 'last'. Default: 'best'."
    )

    parser.add_argument(
        "-w", "--weights", type=str, default=None,
        help="Path to model weights. Required for training 'mlm_regression' (pre-trained MLM) or external data selection in 'influence'/'similarity' modes."
    )

    parser.add_argument(
        "-p", "--peft-type", choices=["biffit", "lora", "ia3"], default=None,
        help="Type of Parameter-Efficient Fine-Tuning (PEFT). If freezing is applied, this overrides it. Options: 'biffit', 'lora', 'ia3'. Default: None."
    )

    parser.add_argument(
        "-r", "--rank", type=int, default=DEFAULT_RANK,
        help=f"LoRA rank value (only used if --peft-type is 'lora'). Default: {DEFAULT_RANK}."
    )

    args = parser.parse_args()

    # Convert "None" strings to actual None
    if args.peft_type == "None":
        args.peft_type = None

    # Validate weight requirement based on mode
    if args.mode in ["influence", "similarity"] and args.weights is None:
        parser.error(f"The --weights argument is required for mode '{args.mode}'.")

    return args

def main():
    args = parse_args()

if __name__ == '__main__':
    main()
