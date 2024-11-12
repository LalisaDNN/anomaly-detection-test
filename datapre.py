from transformers import BertTokenizer
import torch

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_mask(sentence, mlm_prob=0.15):
    """
    Tokenize the sentence and apply MLM masking.
    """
    # Tokenize and get input IDs and attention mask
    tokens = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].squeeze(0)  # Shape: (max_length,)
    attention_mask = tokens["attention_mask"].squeeze(0)  # Shape: (max_length,)
    
    # Apply MLM masking
    mlm_labels = input_ids.clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < mlm_prob) & (input_ids != tokenizer.pad_token_id) & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
    mlm_labels[~mask_arr] = -100  # Set non-masked tokens to -100 to ignore them in the MLM loss
    input_ids[mask_arr] = tokenizer.mask_token_id  # Replace masked tokens with [MASK] token ID
    
    return input_ids, attention_mask, mlm_labels

from torch.utils.data import Dataset, DataLoader
import random

class LogDataset(Dataset):
    def __init__(self, logs, sequence_length, mlm_prob=0.15):
        """
        logs: List of lists, where each sublist is a sequence of log sentences.
        sequence_length: Number of sentences in each log sequence for the parent encoder.
        mlm_prob: Probability of masking a token in MLM task.
        """
        self.logs = logs
        self.sequence_length = sequence_length
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        # Get a sequence of log sentences
        log_sequence = self.logs[idx]
        
        # Prepare child encoder data
        input_ids_list, attention_mask_list, mlm_labels_list = [], [], []
        for sentence in log_sequence:
            input_ids, attention_mask, mlm_labels = tokenize_and_mask(sentence, self.mlm_prob)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            mlm_labels_list.append(mlm_labels)
        
        # Stack to create tensor of shape (sequence_length, max_length)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        mlm_labels = torch.stack(mlm_labels_list)
        
        # Parent encoder coherence label
        coherence_label = self.get_coherence_label(log_sequence)
        
        return input_ids, attention_mask, mlm_labels, coherence_label

    def get_coherence_label(self, log_sequence):
        """
        Generate a label for coherence prediction: 1 if one or more sentences have been replaced, 0 otherwise.
        """
        # Here we define a simple rule to randomly decide coherence for the task; 
        # in practice, you'd have actual coherence labels.
        return torch.tensor(random.choice([0, 1]), dtype=torch.float)  # Example binary label

        
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids, attention_masks, mlm_labels, coherence_labels = zip(*batch)
    
    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    mlm_labels = pad_sequence(mlm_labels, batch_first=True, padding_value=-100)
    
    # Stack coherence labels
    coherence_labels = torch.stack(coherence_labels)
    
    return input_ids, attention_masks, mlm_labels, coherence_labels

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
