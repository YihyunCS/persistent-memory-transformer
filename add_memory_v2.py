import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
from models import RevenaHybrid

class RevenaHybridWithMemoryV2(RevenaHybrid):
    """Enhanced RevenaHybrid with FAISS-based LTM capabilities"""
    
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, 
                 context_length=128, ltm_size=1024):  # Increased memory size
        super().__init__(vocab_size, n_embd, n_head, n_layer, context_length, ltm_size)
        
        # Memory parameters
        self.surprisal_threshold = 4.0  # Slightly lower threshold
        self.cosine_threshold = 0.7  # Adjusted for better recall
        self.batch_size_ltm = 32  # Batch size for LTM queries
        
        # Initialize FAISS index (CPU)
        self.cpu_index = faiss.IndexFlatIP(n_embd)
        self.index = self.cpu_index
        # Attempt to move FAISS index to GPU if CUDA and FAISS GPU is installed
        if torch.cuda.is_available():
            try:
                from faiss import StandardGpuResources, index_cpu_to_gpu
                gpu_res = StandardGpuResources()
                self.index = index_cpu_to_gpu(gpu_res, 0, self.cpu_index)
            except Exception:
                # GPU FAISS unavailable, continue with CPU index
                self.index = self.cpu_index
        self.ltm_memory = None
        self.ltm_enabled = False
        self.memory_counter = 0  # Track number of items in memory
        
    def _normalize_embeddings(self, x):
        """Normalize embeddings for cosine similarity"""
        return F.normalize(x, p=2, dim=-1)
    
    def _process_ltm_batch(self, x_batch):
        """Process a batch of queries against LTM"""
        B, T, C = x_batch.shape
        # Flatten batch for search
        x_flat = x_batch.reshape(-1, C)
        # Normalize for cosine similarity
        x_flat_norm = self._normalize_embeddings(x_flat)
        # Convert to numpy for FAISS
        x_norm = x_flat_norm.detach().cpu().numpy().astype('float32')
        # If there is no memory stored yet, return original
        if self.memory_counter == 0:
            return x_batch
        # FAISS search: get top-k indices and similarities
        k = 5
        D, I = self.index.search(x_norm, k)
        # Convert to torch
        device = x_batch.device
        I_t = torch.from_numpy(I).to(device)
        D_t = torch.from_numpy(D).to(device)
        # Mask out low-similarity entries by setting to large negative before softmax
        D_masked = D_t.clone()
        D_masked[D_masked <= self.cosine_threshold] = float('-1e9')
        # Compute weights
        weights = F.softmax(D_masked, dim=1)  # shape: (B*T, k)
        # Load memory tensor
        mem_t = torch.from_numpy(self.ltm_memory.astype('float32')).to(device)  # shape: (M, C)
        # Gather top-k memory embeddings
        neighbors = mem_t[I_t]  # shape: (B*T, k, C)
        # Weighted sum
        retrieved_flat = (neighbors * weights.unsqueeze(-1)).sum(dim=1)  # shape: (B*T, C)
        # Reshape to batch
        retrieved = retrieved_flat.view(B, T, C)
        # Combine and return
        return x_batch + retrieved
    
    def _update_ltm(self, x, surprisal):
        """Update LTM memory using FAISS"""
        B, T, C = x.shape
        device = x.device
        
        # Find high-surprisal tokens
        mask = surprisal > self.surprisal_threshold
        to_store = x[mask].cpu().numpy().astype('float32')
        
        if len(to_store) > 0:
            # Normalize new memories
            to_store_norm = self._normalize_embeddings(
                torch.from_numpy(to_store)
            ).numpy().astype('float32')
            
            # Initialize or extend memory
            if self.ltm_memory is None:
                self.ltm_memory = to_store
                self.index.add(to_store_norm)
                self.memory_counter = len(to_store)
            else:
                # FIFO eviction if needed
                available_space = self.ltm_size - self.memory_counter
                if available_space < len(to_store):
                    # Remove oldest entries
                    remove_count = len(to_store) - available_space
                    self.ltm_memory = np.concatenate([
                        self.ltm_memory[remove_count:],
                        to_store
                    ])
                    # Reset FAISS index with updated memory
                    self.index.reset()
                    self.index.add(
                        self._normalize_embeddings(
                            torch.from_numpy(self.ltm_memory)
                        ).numpy().astype('float32')
                    )
                else:
                    # Just append
                    self.ltm_memory = np.concatenate([self.ltm_memory, to_store])
                    self.index.add(to_store_norm)
                
                self.memory_counter = len(self.ltm_memory)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens and positions
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # LTM processing if enabled
        if self.ltm_enabled and self.memory_counter > 0:
            # Process in batches to avoid memory issues
            x_batched = torch.split(x, self.batch_size_ltm)
            x = torch.cat([self._process_ltm_batch(batch) for batch in x_batched], dim=0)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss and update memory if training
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1))
            
            if self.ltm_enabled and self.training:
                with torch.no_grad():
                    # Calculate surprisal
                    probs = F.softmax(logits, dim=-1)
                    surprisal = -torch.log2(
                        probs.gather(-1, targets.unsqueeze(-1))
                    ).squeeze(-1)
                    
                    # Update memory
                    self._update_ltm(x, surprisal)
        
        return logits, loss
    
    def enable_ltm(self, enable=True):
        """Enable/disable LTM memory"""
        self.ltm_enabled = enable
        if not enable:
            self.clear_memory()
    
    def clear_memory(self):
        """Clear LTM memory"""
        self.ltm_memory = None
        self.memory_counter = 0
        # Reset FAISS index
        self.index.reset()

def add_memory_v2_to_model(model):
    """Convert a standard RevenaHybrid model to use enhanced memory"""
    return RevenaHybridWithMemoryV2(
        vocab_size=model.token_embedding.num_embeddings,
        n_embd=model.token_embedding.embedding_dim,
        n_head=model.n_head,
        n_layer=len(model.blocks),
        context_length=model.context_length,
        ltm_size=1024  # Increased memory size for V2
    ) 