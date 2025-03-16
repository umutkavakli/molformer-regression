import torch
from tqdm import tqdm

def test(model, test_dataloader, criterion, model_type, device):
    """
    Evaluate model performance on a test dataset.
    
    Args:
        model: Trained model to evaluate
        test_dataloader: DataLoader for test data
        criterion: Loss function for evaluation
        model_type: Type of model ("mlm" or other for regression)
        device: Device to run evaluation on (CPU or GPU)
        
    Returns:
        float: Average test loss
    """
    model.eval()
    test_loss = 0.0

    # test loop with progress bar
    test_progress_bar = tqdm(test_dataloader, desc=f"Test", leave=True)

    with torch.no_grad():
        for batch in test_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            if model_type == "mlm":
                outputs = model(**batch)
                loss = outputs.loss
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
                loss = criterion(outputs, batch["labels"])

            test_loss += loss.item()

            test_progress_bar.set_postfix({"Test Loss": loss.item()})
        
        avg_test_loss = test_loss / len(test_dataloader)
        return avg_test_loss