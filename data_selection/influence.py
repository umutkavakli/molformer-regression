import torch
import pandas as pd
from torch.autograd import grad
from torch.utils.data import DataLoader


def compute_s_test(model, criterion, train_dataset, test_dataset, device):
    """
    Compute the test-side influence vector (s_test) by averaging the gradients of the loss
    on the test dataset, then estimating the inverse Hessian-vector product (IHVP) using the LiSSA algorithm.

    Args:
        model: The PyTorch model.
        criterion: The loss function.
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        device: The device to run the computations on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: The estimated s_test vector.
    """
    grads_total = 0.0
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for batch in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
        loss = criterion(outputs, batch["labels"])
        test_grads = grad(loss, model.parameters())
        grads_total += torch.cat([g.view(-1) for g in test_grads])
    
    avg_test_grad = grads_total / len(test_dataloader)
    return lissa_inverse_hvp(model, criterion, train_dataset, avg_test_grad, device)


def lissa_inverse_hvp(
        model, 
        criterion, 
        train_dataset, 
        v, 
        device,
        batch_size=1,
        scale=1e3,
        damping=0.01,
        repeats=5,
        recursion_depth=500
    ):
    """
    Estimate the inverse Hessian-vector product (IHVP) using the LiSSA (Linear Stochastic Second-order Approximation) algorithm.

    Args:
        model: The PyTorch model.
        criterion: The loss function.
        train_dataset: The training dataset.
        v: The vector to compute the IHVP for (e.g., average test gradients).
        device: The device to run the computations on (e.g., 'cuda' or 'cpu').
        batch_size: The batch size for sampling the training dataset.
        scale: Scaling factor for stability.
        damping: Damping factor to prevent numerical instability.
        repeats: Number of times to repeat the estimation.
        recursion_depth: The number of recursion steps.

    Returns:
        torch.Tensor: The estimated inverse Hessian-vector product.
    """
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ihvp = torch.zeros_like(v).to(device)

    for _ in range(repeats):
        h_estimate = v.clone().to(device)
        iterator = iter(dataloader)

        for _ in range(recursion_depth):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            batch = {key: value.to(device) for key, value in batch.items()} 
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
            loss = criterion(outputs, batch["labels"])

            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flatten_grads = torch.cat([g.view(-1) for g in grads])
            
            hvp = torch.autograd.grad(flatten_grads, model.parameters(), grad_outputs=h_estimate, retain_graph=True)
            hvp = torch.cat([g.contiguous().view(-1) for g in hvp])

            with torch.no_grad():
                hvp = hvp + damping * h_estimate
                h_estimate = v + h_estimate - hvp / scale

        ihvp += h_estimate / scale

    return ihvp / repeats

def computer_influence(model, criterion, train_dataset, test_dataset, device):
    """
    Compute the influence score of each training sample on the test loss using influence functions.

    Args:
        model: The PyTorch model.
        criterion: The loss function.
        train_dataset: The training dataset.
        test_dataset: The test dataset.
        device: The device to run the computations on (e.g., 'cuda' or 'cpu').

    Returns:
        list: A list of influence scores for each training sample.
    """
    s_test = compute_s_test(model, criterion, train_dataset, test_dataset, device)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    scores = []
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()} 
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).squeeze()
        loss = criterion(outputs, batch["labels"])
        train_grads = grad(loss, model.parameters())
        flat_train_grads = torch.cat([g.view(-1) for g in train_grads])

        s = -torch.dot(flat_train_grads, s_test) / len(flat_train_grads)  # scaling the values with dimension size
        scores.append(s.item())
        
        print(f"Completed {i+1}/{len(train_loader)}")
    
    return scores

def best_influences(original_df, scores, influence_type="positive", top_k=None, save_df=True):
    """
    Extract the most influential training samples based on the computed influence scores.

    Args:
        original_df: The original DataFrame containing the training data.
        scores: The list of influence scores.
        influence_type: The type of influence to filter ('positive' or 'negative').
        top_k: The number of top influential samples to return.
        save_df: Whether to save the results to a CSV file.

    Returns:
        list: A list of dictionaries with SMILES, Label, and influence score.
    """
    result = []
    for i, score in enumerate(scores):
        result.append(
            {
                'SMILES': original_df.iloc[i]['SMILES'],
                'Label': original_df.iloc[i]['Label'],
                'influence': score
            }
        )

    result = [item for item in result if item['influence'] >= 0] if influence_type == "positive" else result
    result = sorted(result, key=lambda x: x['influence'], reverse=True)
    
    if top_k:
        result = result[:top_k]
    df = pd.DataFrame(result)

    if save_df:
        df.to_csv("external_dataset_with_influence.csv")
    
    return result