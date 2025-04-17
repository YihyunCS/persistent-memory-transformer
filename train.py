import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from models import create_RevenaHybridTiny_70M
from add_memory import add_memory_to_model
import os
import time
import numpy as np
from tqdm import tqdm
import wandb

def get_batch(data, batch_size, context_length, device):
    """Generate input and target batches from data"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_length+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-ltm', action='store_true', help='Enable LTM memory')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model (no LTM)')
    parser.add_argument('--data', type=str, default='data/train.bin', help='Training data file')
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.1, help='Weight decay')
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="revena-hybrid", config=vars(args))
    wandb.config.update({"LTM_enabled": args.with_ltm})

    # Load data
    data = np.fromfile(args.data, dtype=np.uint16)
    vocab_size = 50257  # GPT-2 vocab size

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_RevenaHybridTiny_70M(vocab_size).to(device)
    
    if args.with_ltm:
        model = add_memory_to_model(model).to(device)
        model.enable_ltm(True)
        save_dir = "trained/with_ltm"
    else:
        save_dir = "trained/baseline"
    
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
        
        log_data = {
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/tokens_per_sec": tokens_per_sec,
        }
        
        if args.with_ltm:
            # TODO: Add LTM-specific metrics
            pass
            
        wandb.log(log_data, step=step)
        
        # Save checkpoint
        if (step + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"{save_dir}/step_{step+1}.pt")

    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/final.pt")

if __name__ == "__main__":
    main()