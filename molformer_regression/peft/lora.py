import torch 
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Implementation of Low-Rank Adaptation (LoRA) for transformer layers.
    
    LoRA reduces the number of trainable parameters by using low-rank decomposition 
    to approximate weight updates. Instead of directly updating the pre-trained weights W,
    it adds a low-rank decomposition BA where B and A are trainable matrices.
    """
    def __init__(self, W, rank, alpha):
        """
        Initialize the LoRA layer.
        
        Args:
            W: The original weight matrix/layer to be adapted
            rank: The rank of the low-rank decomposition (smaller = fewer parameters)
            alpha: Scaling factor that affects the magnitude of the adaptation
        """
        super().__init__()
        in_features, out_features = W.in_features, W.out_features
        
        self.W = W
        self.B = nn.Parameter(torch.zeros(rank, out_features), requires_grad=True)
        self.A = nn.Parameter(torch.zeros(in_features, rank), requires_grad=True)
        self.scale = alpha / rank
        
        # weight matrix of A initialized with normal distribution
        nn.init.normal_(self.A, mean=0, std=1)

    def forward(self, x):
        """
        Forward pass combining the original transformation with the low-rank adaptation.
        
        Formula: output = W(x) + (alpha/rank) * (x @ A @ B)
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Transformed output with LoRA adaptation applied
        """
        # W + (alpha/rank) * BA 
        return self.W(x) + self.scale * (torch.matmul(torch.matmul(x, self.A), self.B))