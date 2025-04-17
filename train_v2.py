import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from models import create_RevenaHybridTiny_70M
from add_memory_v2 import add_memory_v2_to_model
import os
import time
import numpy as np
from tqdm import tqdm
import wandb
import json

def get_batch(data, batch_size, context_length, device):
    """Generate input and target batches from data"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_length+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train.bin', help='Training data file')
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight decay')
    args = parser.parse_args()

    # Initialize wandb with a new project name
    wandb.init(project="revena-hybrid-v2", config=vars(args))
    wandb.config.update({"LTM_version": "v2"})

    # Load data
    data = np.fromfile(args.data, dtype=np.uint16)
    vocab_size = 50257  # GPT-2 vocab size

    # Create model with V2 memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = create_RevenaHybridTiny_70M(vocab_size).to(device)
    model = add_memory_v2_to_model(base_model).to(device)
    model.enable_ltm(True)
    save_dir = "trained/with_ltm_v2"
    
    os.makedirs(save_dir, exist_ok=True)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    
    # Warmup scheduler
    warmup_steps = 500
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
    # Cosine scheduler after warmup
    cosine_steps = max(args.steps - warmup_steps, 1)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup, cosine], 
        milestones=[warmup_steps]
    )

    # Training loop
    model.train()
    start_time = time.time()
    tokens_processed = 0
    memory_hit_rates = []
    
    for step in tqdm(range(args.steps), desc="Training"):
        # Get batch
        x, y = get_batch(data, args.batch_size, model.context_length, device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        tokens_processed += x.numel()
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_processed / elapsed
        
        # Calculate memory hit rate
        hit_rate = 0.0
        if model.memory_counter > 0:
            with torch.no_grad():
                # Get embeddings
                tok_emb = model.token_embedding(x)
                pos_emb = model.position_embedding(torch.arange(x.size(1), device=x.device))
                hidden = tok_emb + pos_emb
                
                # Process through transformer blocks
                for block in model.blocks:
                    hidden = block(hidden)
                
                # Apply layer norm and reshape
                x_flat = model.ln_f(hidden).reshape(-1, model.n_embd)
                x_norm = model._normalize_embeddings(x_flat).cpu().numpy().astype('float32')
                D, _ = model.index.search(x_norm, k=1)
                hit_rate = (D > model.cosine_threshold).mean().item()
                memory_hit_rates.append(hit_rate)
        
        log_data = {
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/tokens_per_sec": tokens_per_sec,
            "train/memory_size": model.memory_counter,
            "train/memory_hit_rate": hit_rate
        }
            
        wandb.log(log_data, step=step)
        
        # Save checkpoint
        if (step + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"{save_dir}/step_{step+1}.pt")

    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/final.pt")
    
    # Save training statistics
    stats = {
        "final_loss": loss.item(),
        "tokens_per_sec": tokens_per_sec,
        "memory_size": model.memory_counter,
        "avg_memory_hit_rate": np.mean(memory_hit_rates),
        "final_memory_hit_rate": hit_rate
    }
    
    with open(f"{save_dir}/training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main() 