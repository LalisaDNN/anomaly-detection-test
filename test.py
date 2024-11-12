import random
from typing import List, Tuple

def clean_paragraph(paragraph: str) -> str:
    # Remove "<br /><br />" tokens from the paragraph
    return paragraph.replace("<br /><br />", " ")

def split_into_sentences(paragraph: str) -> List[str]:
    # Split a paragraph into sentences based on ". "
    sentences = paragraph.strip().split(". ")
    return [s.strip() for s in sentences if s]

def create_windows(sentences: List[str], window_size: int, stride: int) -> List[List[str]]:
    # Create windows with a specified size and stride
    windows = []
    for i in range(0, len(sentences), stride):
        window = sentences[i:i + window_size]
        # Pad the window if it's shorter than the window_size
        if len(window) < window_size:
            padding = ["[PAD]"] * (window_size - len(window))
            window.extend(padding)
        windows.append(window)
    return windows

def replace_with_random_sentences(window: List[str], corpus_sentences: List[str], replace_count: int) -> List[str]:
    # Replace `replace_count` sentences in the window with random sentences from the corpus
    modified_window = window[:]
    replace_indices = random.sample(range(len(window)), replace_count)
    for idx in replace_indices:
        if modified_window[idx] != "[PAD]":  # Only replace non-padding sentences
            random_sentence = random.choice(corpus_sentences)
            modified_window[idx] = random_sentence
    return modified_window

def generate_data(paragraphs: List[str], window_size: int, stride: int) -> Tuple[List[List[str]], List[int], List[List[int]]]:
    all_windows = []
    labels = []
    attention_masks = []
    
    # Preprocess paragraphs and gather all sentences for random replacement
    cleaned_paragraphs = [clean_paragraph(paragraph) for paragraph in paragraphs]
    all_sentences = [sentence for paragraph in cleaned_paragraphs for sentence in split_into_sentences(paragraph)]
    
    for paragraph in cleaned_paragraphs:
        sentences = split_into_sentences(paragraph)
        windows = create_windows(sentences, window_size, stride)
        
        for window in windows:
            # Decide if this window should have replacements
            replace_flag = random.choice([0, 1])  # 0 for no replacement, 1 for replacement
            
            if replace_flag == 1:
                # Replace 1 or 2 sentences in the window with random sentences from the corpus
                replace_count = random.choice([1, 2])
                modified_window = replace_with_random_sentences(window, all_sentences, replace_count)
                label = 1
            else:
                modified_window = window
                label = 0
            
            # Generate attention mask: 1 for valid (non-padding) sentences, 0 for padding sentences
            attention_mask = [1 if sentence != "[PAD]" else 0 for sentence in modified_window]

            all_windows.append(modified_window)
            labels.append(label)
            attention_masks.append(attention_mask)

    return all_windows, labels, attention_masks

# Example usage
paragraphs = [
    "This is the first sentence.<br /><br />Here is the second one. And now the third sentence. Finally, the fourth sentence.",
    "Another paragraph begins here.<br /><br />It has multiple sentences. The third sentence is here. And it ends with this one."
]
window_size = 3
stride = 2

data, labels, attention_masks = generate_data(paragraphs, window_size, stride)

# Display the prepared data
for i in range(len(data)):
    print("Window:", data[i])
    print("Label:", labels[i])
    print("Attention Mask:", attention_masks[i])
    print()
