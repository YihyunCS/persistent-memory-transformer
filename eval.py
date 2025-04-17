import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from models import create_RevenaHybridTiny_70M
from add_memory import add_memory_to_model
from tqdm import tqdm

def get_batch(data, batch_size, context_length, device):
    """Generate input and target batches from data"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_length+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def calculate_bpc(model, data, batch_size=8, num_batches=100):
    """Calculate bits per character (BPC)"""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Calculating BPC"):
            x, y = get_batch(data, batch_size, model.context_length, device)
            logits, loss = model(x, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
    
    return total_loss / total_tokens / np.log(2)

def long_range_recall_test(model, num_tests=100):
    """Synthetic test for long-range dependency recall"""
    if not model.ltm_enabled:
        return 0.0
        
    device = next(model.parameters()).device
    vocab_size = model.token_embedding.num_embeddings
    recall = 0
    
    for _ in range(num_tests):
        # Create a random pattern to remember
        pattern = torch.randint(0, vocab_size, (16,), device=device)
        
        # Process pattern through model to store in LTM
        model.clear_memory()
        model(torch.cat([pattern, torch.zeros(model.context_length-16, device=device).long()]).unsqueeze(0))
        
        # Create test sequence with pattern at beginning and cue at end
        test_seq = torch.cat([
            pattern,
            torch.randint(0, vocab_size, (model.context_length-32,), device=device),
            pattern[:1]  # Cue
        ])
        
        # Get model's prediction for next token
        logits, _ = model(test_seq.unsqueeze(0))
        pred = logits[0, -1].argmax()
        
        # Check if prediction matches the next token in pattern
        if pred == pattern[1]:
            recall += 1
    
    return recall / num_tests

def evaluate_model(model, val_data):
    """Evaluate model and return metrics"""
    device = next(model.parameters()).device
    
    # Time inference speed
    x = torch.randint(0, 100, (1, model.context_length), device=device)
    start_time = time.time()
    for _ in range(100):
        model(x)
    elapsed = time.time() - start_time
    tokens_per_sec = (100 * model.context_length) / elapsed
    
    # Calculate metrics
    bpc = calculate_bpc(model, val_data)
    lr_recall = long_range_recall_test(model) if model.ltm_enabled else 0.0
    hit_rate = 0.0  # TODO: Track actual LTM hit rate during evaluation
    
    return {
        "bpc": bpc,
        "long_range_recall": lr_recall,
        "memory_hit_rate": hit_rate,
        "tokens_per_sec": tokens_per_sec
    }

def main():
    # Load validation data
    val_data = np.fromfile("data/val.bin", dtype=np.uint16)
    vocab_size = 50257  # GPT-2 vocab size
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate baseline model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = create_RevenaHybridTiny_70M(vocab_size).to(device)
    baseline.load_state_dict(torch.load("trained/baseline/final.pt"))
    results["baseline"] = evaluate_model(baseline, val_data)
    
    # Evaluate LTM model
    ltm_model = add_memory_to_model(create_RevenaHybridTiny_70M(vocab_size).to(device))
    ltm_model.load_state_dict(torch.load("trained/with_ltm/final.pt"))
    ltm_model.enable_ltm(True)
    results["with_ltm"] = evaluate_model(ltm_model, val_data)
    
    # Save results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()