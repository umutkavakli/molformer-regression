import torch
import torch.nn as nn


# specify model with a regression head
class MoLFormerWithRegressionHead(nn.Module):
    """
    A model class that adds a regression head on top of a MoLFormer base model.
    
    This architecture uses a pre-trained MoLFormer model for molecular representation
    and adds a regression head to predict continuous values (e.g., solubility, toxicity).
    Supports different freezing strategies for transfer learning.
    """
    def __init__(self, base_model, dropout_rate=0.2, is_mlm=False, freeze=None):
        """
        Initialize the MoLFormer with regression head.
        
        Args:
            base_model: Pre-trained MoLFormer model
            dropout_rate: Dropout probability in the regression head (default: 0.2)
            is_mlm: Whether the base_model is a masked language model (default: False)
            freeze: Freezing strategy - "none", "full", or "partial" (default: "none")
        """
        super(MoLFormerWithRegressionHead, self).__init__()
        self.hidden_size = base_model.config.hidden_size

        if is_mlm:
            self.base_model = base_model.molformer
        else:
            self.base_model = base_model

        if freeze == None:
          for param in self.base_model.parameters():
            param.requires_grad = True
        elif freeze == "full":
          for param in self.base_model.parameters():
            param.requires_grad = False
        elif freeze == "partial":
          for name, param in self.base_model.named_parameters():
            if (
              "encoder.layer.11" in name or
              "encoder.layer.10" in name or
              "encoder.layer.9" in name
            ):
              param.requires_grad = True
            else:
              param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.GELU(), 
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of the input SMILES strings
            attention_mask: Attention mask for the input tokens
            
        Returns:
            torch.Tensor: Predicted regression values
        """
        h = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        h = torch.mean(h.last_hidden_state, dim=1) # [batch, sequence_length, hidden_dimension] --> [batch, hidden_dimension] 
        logits = self.regressor(h)
        
        return logits
    