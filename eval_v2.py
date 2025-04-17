import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from models import create_RevenaHybridTiny_70M
from add_memory_v2 import add_memory_v2_to_model
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

def long_range_recall_test(model, num_tests=100, pattern_length=32):
    """Enhanced synthetic test for long-range dependency recall"""
    if not model.ltm_enabled:
        return 0.0
        
    device = next(model.parameters()).device
    vocab_size = model.token_embedding.num_embeddings
    recall = 0
    
    for _ in range(num_tests):
        # Create a longer random pattern to remember
        pattern = torch.randint(0, vocab_size, (pattern_length,), device=device)
        
        # Process pattern through model to store in LTM
        model.clear_memory()
        model(torch.cat([
            pattern, 
            torch.zeros(model.context_length-pattern_length, device=device).long()
        ]).unsqueeze(0))
        
        # Create test sequence with pattern at beginning and cue at end
        # Add more random tokens in between for a harder test
        test_seq = torch.cat([
            pattern,
            torch.randint(0, vocab_size, (model.context_length-pattern_length-1,), device=device),
            pattern[:1]  # Cue
        ])
        
        # Get model's prediction for next token
        logits, _ = model(test_seq.unsqueeze(0))
        
        # Check top-k predictions
        k = 5
        topk_preds = logits[0, -1].topk(k).indices
        if pattern[1] in topk_preds:
            recall += 1
    
    return recall / num_tests

def memory_retrieval_test(model, num_tests=100):
    """Test memory retrieval quality"""
    if not model.ltm_enabled or model.memory_counter == 0:
        return 0.0
    
    device = next(model.parameters()).device
    total_similarity = 0
    
    with torch.no_grad():
        # Sample random memories
        for _ in range(num_tests):
            idx = np.random.randint(0, model.memory_counter)
            query = torch.from_numpy(model.ltm_memory[idx]).to(device)
            
            # Get normalized query
            query_norm = model._normalize_embeddings(query.unsqueeze(0))
            
            # Search in FAISS
            D, I = model.index.search(query_norm.cpu().numpy().astype('float32'), k=2)
            
            # Calculate similarity to closest non-identical memory
            # Skip first result as it's the query itself
            total_similarity += D[0][1]
    
    return total_similarity / num_tests

def evaluate_model_v2(model, val_data):
    """Evaluate model with enhanced metrics"""
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
    lr_recall = long_range_recall_test(model)
    retrieval_quality = memory_retrieval_test(model)
    
    # Memory statistics
    memory_stats = {
        "size": model.memory_counter,
        "capacity": model.ltm_size,
        "utilization": model.memory_counter / model.ltm_size if model.ltm_size > 0 else 0
    }
    
    return {
        "bpc": bpc,
        "long_range_recall": lr_recall,
        "retrieval_quality": retrieval_quality,
        "tokens_per_sec": tokens_per_sec,
        "memory_stats": memory_stats
    }

def main():
    # Load validation data
    val_data = np.fromfile("data/val.bin", dtype=np.uint16)
    vocab_size = 50257  # GPT-2 vocab size
    
    # Initialize results dictionary
    results = {}
    
    # Load previous results for comparison
    try:
        with open("results.json", "r") as f:
            prev_results = json.load(f)
            results.update(prev_results)
    except FileNotFoundError:
        pass
    
    # Evaluate LTM V2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = create_RevenaHybridTiny_70M(vocab_size).to(device)
    ltm_v2_model = add_memory_v2_to_model(base_model).to(device)
    ltm_v2_model.load_state_dict(torch.load("trained/with_ltm_v2/final.pt"))
    ltm_v2_model.enable_ltm(True)
    results["with_ltm_v2"] = evaluate_model_v2(ltm_v2_model, val_data)
    
    # Save all results
    with open("results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 