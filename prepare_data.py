from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
import os
import tqdm

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load OpenWebText in streaming mode
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    
    # Take first 10,000 samples
    dataset = dataset.take(10000)
    
    # Process documents with sliding window
    window_size = 128
    stride = 64
    train_tokens = []
    val_tokens = []
    
    for doc in tqdm.tqdm(dataset, desc="Processing documents"):
        # Tokenize document
        tokens = tokenizer.encode(doc["text"])
        
        # Skip empty documents
        if len(tokens) < window_size:
            continue
            
        # Apply sliding window
        for i in range(0, len(tokens) - window_size + 1, stride):
            chunk = tokens[i:i + window_size]
            
            # Randomly assign to train or val (90/10 split)
            if np.random.random() < 0.9:
                train_tokens.extend(chunk)
            else:
                val_tokens.extend(chunk)
    
    # Convert to numpy arrays
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    
    # Save to binary files
    train_tokens.tofile("data/train.bin")
    val_tokens.tofile("data/val.bin")
    
    print(f"Saved {len(train_tokens)} tokens to data/train.bin")
    print(f"Saved {len(val_tokens)} tokens to data/val.bin")

if __name__ == "__main__":
    main()