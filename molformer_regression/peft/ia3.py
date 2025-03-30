import torch 
import torch.nn as nn

class IA3AdapterLayer(nn.Module):
    """
    Implementation of IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
    
    IA³ is a parameter-efficient fine-tuning method that applies learnable scaling vectors
    to transformer activations. It's even more parameter-efficient than LoRA.
    """
    def __init__(self, W, is_ff=False):
        """
        Initialize the IA³ adapter layer.
        
        Args:
            W: The original weight matrix/layer to be adapted
            is_ff: Boolean indicating if this is a feed-forward layer (True) or attention layer (False)
        """
        super().__init__()
        self.is_ff = is_ff
        if is_ff:
            dim = W.in_features
        else:
            dim = W.out_features

        self.W = W
        self.scale_vec = nn.Parameter(torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        Forward pass applying the IA³ scaling.
        
        For attention layers: scales the output (W(x) * scale_vector)
        For feed-forward layers: scales the input (W(x * scale_vector))
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Transformed output with IA³ adaptation applied
        """
        # apply scale factor for dense just before next dense (l_ff . s(W_1x))W_2
        return self.W(x) * self.scale_vec if not self.is_ff else self.W(x * self.scale_vec)