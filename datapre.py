import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class LogSequenceDataset(Dataset):
    def __init__(self, data, labels, window_size, max_len, tokenizer_name="bert-base-uncased"):
        """
        Args:
            data: List[List[str]] - Each sublist contains sentences for a single window (sequence).
            labels: List[int] - Label for each window (1 for incoherent, 0 for coherent).
            window_size: int - Number of sentences in each window.
            max_len: int - Maximum token length for each sentence.
            tokenizer_name: str - Name of the BERT tokenizer to use.
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]  # A list of sentences in the current window
        label = self.labels[idx]  # Label for this window

        # Initialize lists to hold tokenized data for the window
        input_ids = []
        attention_masks = []

        # Tokenize each sentence in the window
        for sentence in window:
            encoded = self.tokenizer(
                sentence,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids.append(encoded["input_ids"].squeeze())  # Shape: (max_len,)
            attention_masks.append(encoded["attention_mask"].squeeze())  # Shape: (max_len,)

        # Stack tokenized sentences to get window tensors
        input_ids = torch.stack(input_ids)  # Shape: (window_size, max_len)
        attention_masks = torch.stack(attention_masks)  # Shape: (window_size, max_len)

        return {
            "input_ids": input_ids,             # Shape: (window_size, max_len)
            "attention_mask": attention_masks,  # Shape: (window_size, max_len)
            "label": torch.tensor(label, dtype=torch.long)
        }

# Instantiate dataset
window_size = 5   # Example window size
max_len = 50      # Maximum token length for each sentence
data = [["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"], ...]  # Example data
labels = [0, 1, ...]  # Example labels

dataset = LogSequenceDataset(data, labels, window_size, max_len)

# Use DataLoader for batching
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example of iterating through data_loader
for batch in data_loader:
    input_ids = batch["input_ids"]            # Shape: (batch_size, window_size, max_len)
    attention_mask = batch["attention_mask"]  # Shape: (batch_size, window_size, max_len)
    labels = batch["label"]                   # Shape: (batch_size,)
    # Now `input_ids`, `attention_mask`, and `labels` are ready for model input
