import torch
import torch.nn as nn
import torch.nn.functional as F
from models import RevenaHybrid
from torch.nn import TransformerEncoderLayer
import math

class RevenaHybridWithMemory(RevenaHybrid):
    """Extends RevenaHybrid with full STM and LTM capabilities"""
    
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, 
                 context_length=128, ltm_size=256):
        super().__init__(vocab_size, n_embd, n_head, n_layer, context_length, ltm_size)
        
        # Memory parameters
        self.surprisal_threshold = 4.5  # bits
        self.cosine_threshold = 0.8
        self.ltm_ptr = 0  # Pointer for FIFO eviction
        
        # Custom attention mask for sliding window
        self.register_buffer("attn_mask", self._create_sliding_window_mask(context_length))
        
    def _create_sliding_window_mask(self, size):
        """Create attention mask for sliding window attention"""
        mask = torch.full((size, size), float('-inf'))
        for i in range(size):
            for j in range(max(0, i-64), min(size, i+64)):
                mask[i,j] = 0
        return mask
    
    def _calculate_surprisal(self, logits, targets):
        """Calculate token-level surprisal in bits"""
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -torch.log2(target_probs)
    
    def _process_ltm(self, x):
        """Process input with LTM memory"""
        B, T, C = x.shape
        
        # Calculate cosine similarity between input and memory
        if self.ltm_memory is not None and len(self.ltm_memory) > 0:
            mem = self.ltm_memory.unsqueeze(0).expand(B, -1, -1)  # B x M x C
            sim = F.cosine_similarity(x.unsqueeze(2), mem.unsqueeze(1), dim=-1)  # B x T x M
            
            # Retrieve from memory where similarity > threshold
            mask = sim > self.cosine_threshold
            retrieved = torch.zeros_like(x)
            for b in range(B):
                for t in range(T):
                    if mask[b,t].any():
                        # Weighted sum of similar memories
                        weights = sim[b,t][mask[b,t]]
                        weights = weights / weights.sum()
                        retrieved[b,t] = (mem[b][mask[b,t]] * weights.unsqueeze(-1)).sum(0)
            
            # Combine with original input
            x = x + retrieved
        
        return x
    
    def _update_ltm(self, x, surprisal):
        """Update LTM memory based on surprisal"""
        B, T, C = x.shape
        
        for b in range(B):
            for t in range(T):
                if surprisal[b,t] > self.surprisal_threshold:
                    # Store high-surprisal tokens in LTM
                    if self.ltm_memory is None:
                        self.ltm_memory = torch.zeros((self.ltm_size, C), device=x.device)
                    
                    # FIFO eviction if full
                    self.ltm_memory[self.ltm_ptr] = x[b,t]
                    self.ltm_ptr = (self.ltm_ptr + 1) % self.ltm_size
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens and positions
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # STM processing with sliding window attention
        for block in self.blocks:
            x = block(x, src_mask=self.attn_mask[:T,:T])
        
        # Final projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss and surprisal if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1))
            
            # Calculate surprisal and update LTM if enabled
            if self.ltm_enabled:
                surprisal = self._calculate_surprisal(logits, targets)
                self._update_ltm(x, surprisal)
        
        return logits, loss

def add_memory_to_model(model):
    """Convert a standard RevenaHybrid model to use memory"""
    return RevenaHybridWithMemory(
        vocab_size=model.token_embedding.num_embeddings,
        n_embd=model.token_embedding.embedding_dim,
        n_head=model.n_head,
        n_layer=len(model.blocks),
        context_length=model.context_length,
        ltm_size=model.ltm_size
    )