from lora import LoRALayer
from ia3 import IA3AdapterLayer


class PEFT:
    """
    Parameter-Efficient Fine-Tuning (PEFT) manager that applies different PEFT methods
    to a pre-trained model.
    
    Supports BitFit, LoRA, and IA³ fine-tuning methods, which update only a small subset
    of model parameters, enabling efficient adaptation with limited computational resources.
    """
    def __init__(self, model):
        """
        Initialize the PEFT manager. 
        
        By default, freezes all parameters except for the regressor head.
        
        Args:
            model: The pre-trained model to apply PEFT methods to
        """
        self.model = model

        for name, param in self.model.named_parameters():
            if "regressor" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def apply_bitfit(self):
        """
        Apply BitFit fine-tuning method.
        
        BitFit only updates the bias terms of the model while keeping other parameters frozen,
        which is an extremely parameter-efficient approach.
        """
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.requires_grad = True

    def apply_lora(self, rank, alpha):
        """
        Apply LoRA (Low-Rank Adaptation) to the model's attention layers.
        
        Replaces the query, key, and value projections in each transformer layer
        with LoRA-adapted versions.
        
        Args:
            rank: The rank of the low-rank decomposition
            alpha: Scaling factor for the LoRA adaptation
        """
        for block in self.model.base_model.encoder.layer:
            block.attention.self.query = LoRALayer(block.attention.self.query, rank, alpha)
            block.attention.self.key   = LoRALayer(block.attention.self.key, rank, alpha)
            block.attention.self.value = LoRALayer(block.attention.self.value, rank, alpha)

    def apply_ia3(self):
        """
        Apply IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations).
        
        Adds IA³ adapters to the key and value projections in the attention layers
        and to the intermediate dense layer in the feed-forward network.
        """
        for block in self.model.base_model.encoder.layer:
            block.attention.self.key = IA3AdapterLayer(block.attention.self.key)
            block.attention.self.value = IA3AdapterLayer(block.attention.self.value)
            block.intermediate.dense = IA3AdapterLayer(block.intermediate.dense, is_ff=True)