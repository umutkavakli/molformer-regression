import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader


class SiameseSimilarity:
    """
    A class for calculating similarity between embeddings using a siamese network approach.
    Compares embeddings from an external dataset against a validation dataset using cosine similarity.
    
    The similarity score is adjusted based on the difference in labels to prioritize compounds
    with similar properties.
    """
    def __init__(self, model, val_dataset, ext_dataset, df):
        """
        Initialize the SiameseSimilarity class.
        
        Args:
            model: The pre-trained model for generating embeddings
            val_dataset: Validation dataset containing reference compounds
            ext_dataset: External dataset containing compounds to compare
            df: DataFrame containing metadata like SMILES and Label information
        """
        self.model = model
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        self.ext_dataloader = DataLoader(ext_dataset, batch_size=1, shuffle=False)
        self.df = df
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        

    def calculate_similarity(self):
        """
        Calculate similarity scores between compounds in the external dataset and validation dataset.
        
        The method:
        1. Generates embeddings for each compound in both datasets
        2. Computes cosine similarity between embeddings
        3. Adjusts similarity based on label differences
        4. Sorts results by similarity score
        
        Returns:
            DataFrame: Sorted results containing similarity scores, SMILES strings, and labels
        """
        results = []
        self.model.eval()
        with torch.no_grad():
            for i, x_e in enumerate(self.ext_dataloader):
                e_input_ids, e_attention_mask = x_e["input_ids"].unsqueeze(0)[0].to(self.device), x_e["attention_mask"].unsqueeze(0)[0].to(self.device)
                e_label = x_e["labels"].to(self.device)
                e_out = self.model(input_ids=e_input_ids, attention_mask=e_attention_mask)
                e_mean = torch.mean(e_out[0], dim=1)

                total_sim = 0.0
                for x_v in self.val_dataloader:
                    # for batch size 1, some manipulations are required to pass data to the model
                    v_input_ids, v_attention_mask = x_v["input_ids"].unsqueeze(0)[0].to(self.device), x_v["attention_mask"].unsqueeze(0)[0].to(self.device)
                    v_label = x_v["labels"].to(self.device)
                    v_out = self.model(input_ids=v_input_ids, attention_mask=v_attention_mask)
                    v_mean = torch.mean(v_out[0], dim=1)

                    total_sim += (self.cos(e_mean, v_mean) - torch.abs(e_label - v_label))**2
                total_sim = total_sim / len(self.val_dataloader)
                
                results.append(
                    {
                        "similarity": total_sim.item(),
                        "SMILES": self.df["SMILES"][i],
                        "Label": self.df["Label"][i]
                    }
                )
                sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
                result_df = pd.DataFrame(sorted_results).to_csv("similarity_based_results.csv")

        return result_df
