import time
import torch
from tqdm import tqdm

def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, device, model_type, patience=10, save_best_weights=True):
    """
    Train a model with early stopping based on validation loss.
    
    Implements a training loop with progress tracking, early stopping, and model checkpointing.
    Supports both masked language modeling and regression models.
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        epochs: Maximum number of training epochs
        device: Device to run training on (CPU or GPU)
        model_type: Type of model ("mlm" or other for regression)
        patience: Number of epochs to wait for validation loss improvement (default: 10)
        save_best_weights: Whether to save the best model weights (default: True)
        
    Returns:
        dict: Training statistics including losses and training time
    """
    patience_counter = 0
    best_loss = float("inf")

    stats = {
        "losses": {
            "train": [],
            "val": []
        }
    }

    # starting training time
    start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # training loop with progress bar
        train_progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=True)

        for batch in train_progress_bar:
            # getting input ids, attention masks and labels for forward/backward propagation
            batch = {k: v.to(device) for k, v in batch.items()}

            # resetting gradients to zero before backprop
            optimizer.zero_grad()

            # check if mlm or regression model
            if model_type == "mlm":
                outputs = model(**batch)
                loss = outputs.loss
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
                loss = criterion(outputs, batch["labels"])

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # updating training progress bar with current batch loss
            train_progress_bar.set_postfix({"Training Loss": loss.item()})
        
        scheduler.step()
        

        avg_train_loss = train_loss / len(train_dataloader)
        stats["losses"]["train"].append(avg_train_loss)

        # validation loop starts
        model.eval()
        val_loss = 0.0
        
        val_progress_bar = tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}", leave=True)

        with torch.no_grad():
            for batch in val_progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}

                if model_type == "mlm":
                    outputs = model(**batch)
                    loss = outputs.loss
                else:
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
                    loss = criterion(outputs, batch["labels"])

                val_loss += loss.item()

                # update validation progress bar with current batch loss
                val_progress_bar.set_postfix({"Validation Loss": loss.item()})
        
        avg_val_loss = val_loss / len(val_dataloader)
        stats["losses"]["val"].append(avg_val_loss)

        # Print epoch summary
        print(f"\n[INFO]: Epoch {epoch + 1}/{epochs} - Train Loss: {stats['losses']['train'][-1]:.4f}, Validation Loss: {stats['losses']['val'][-1]:.4f}\n")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0

            if save_best_weights:
                torch.save(model.state_dict(), f"./best_{model_type}_model.pt")
        else:
            patience_counter += 1
            print(f"[INFO]: Validation loss does not improve enough, remaining patience: {patience - patience_counter}\n")


        if patience_counter >= patience:
            print(f"[INFO]: Stopping training in epoch {epoch + 1}")
            break
        
    if not save_best_weights:
        torch.save(model.state_dict(), f"./last_{model_type}_model.pt")
    
    # calculate total training time in seconds 
    training_time = time.time() - start
    minutes = int(training_time // 60)
    seconds = int(training_time % 60)

    stats["training_time"] = f"{minutes} mins, {seconds} secs"

    return stats
