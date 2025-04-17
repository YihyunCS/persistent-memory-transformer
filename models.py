import torch
import torch.nn as nn
from torch.nn import functional as F

class RevenaHybrid(nn.Module):
    """Hybrid memory transformer with STM and LTM capabilities"""
    
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, 
                 context_length=128, ltm_size=256):
        super().__init__()
        self.context_length = context_length
        self.ltm_size = ltm_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_length, n_embd)
        
        # Store model config
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        
        # Transformer blocks with sliding window attention (STM)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # LTM memory bank
        self.ltm_memory = None
        self.ltm_enabled = False
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params/1e6:.1f}M")
    
    def forward(self, idx, targets=None):
        # Ensure input is on same device as model
        idx = idx.to(next(self.parameters()).device)
        B, T = idx.shape
        
        # Initialize memory if enabled and not already initialized
        if self.ltm_enabled and self.ltm_memory is None:
            self.ltm_memory = torch.zeros((self.ltm_size, self.n_embd),
                                       device=idx.device)
        
        # Embed tokens and positions
        tok_emb = self.token_embedding(idx)
        if targets is not None:
            targets = targets.to(next(self.parameters()).device)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # STM processing with sliding window attention
        window_size = self.context_length
        for block in self.blocks:
            # Process in sliding windows
            for i in range(0, T, window_size//2):
                window = x[:, i:i+window_size]
                processed = block(window)
                x[:, i:i+window_size] = processed
        
        # LTM processing if enabled
        if self.ltm_enabled and self.ltm_memory is not None:
            x = self._process_ltm(x)
        
        # Final projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 targets.view(-1))
            
            # Calculate surprisal and update LTM if enabled
            if self.ltm_enabled:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    surprisal = -torch.log2(probs.gather(-1, targets.unsqueeze(-1))).squeeze(-1)
                    
                    # Store high-surprisal tokens in LTM (FIFO)
                    high_surprisal = surprisal > 4.5
                    if high_surprisal.any():
                        to_store = x[high_surprisal]
                        if to_store.size(0) > 0:
                            if self.ltm_memory.size(0) + to_store.size(0) > self.ltm_size:
                                excess = (self.ltm_memory.size(0) + to_store.size(0)) - self.ltm_size
                                self.ltm_memory = self.ltm_memory[excess:]
                            self.ltm_memory = torch.cat([self.ltm_memory, to_store], dim=0)
        
        return logits, loss
    
    def _process_ltm(self, x):
        """Process input with LTM memory"""
        # Calculate cosine similarity between input and memory
        x_norm = F.normalize(x, p=2, dim=-1)
        mem_norm = F.normalize(self.ltm_memory, p=2, dim=-1)
        sim = torch.matmul(x_norm, mem_norm.T)
        
        # Retrieve from LTM where similarity > 0.8
        mask = sim > 0.8
        retrieved = torch.zeros_like(x)
        for i in range(mask.size(0)):
            if mask[i].any():
                retrieved[i] = self.ltm_memory[mask[i]].mean(dim=0)
        
        # Combine with original input
        return x + retrieved
    
    def enable_ltm(self, enable=True):
        """Enable/disable LTM memory"""
        self.ltm_enabled = enable
        # Memory initialization now handled in forward()
    
    def clear_memory(self):
        """Clear both STM and LTM memory"""
        self.ltm_memory = None
        # STM is stateless in this implementation

# Create model instances
def create_RevenaHybridTiny_70M(vocab_size):
    """Create 70M parameter hybrid model"""
    return RevenaHybrid(vocab_size, n_embd=512, n_head=8, n_layer=6)

def create_StandardGPT_70M(vocab_size):
    """Create 70M parameter baseline model"""
    return RevenaHybrid(vocab_size, n_embd=512, n_head=8, n_layer=6)