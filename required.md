🧠 persistent-memory transformer (stm + ltm)

research goal: test whether dual memory improves compression and long-range recall

hardware: single rtx 4060 8gb
tokenizer: gpt-2
model size: ~70m params
context length: 128
dataset: openwebtext (10k samples, streaming only)
1. models.py
RevenaHybrid should be a transformer created to have STM and LTM

    RevenaHybridTiny-70M → STM + LTM model 
    Standard GPT-70M

2. prepare_data.py
steps:

    load openwebtext from HuggingFace using streaming mode

    extract 10,000 samples only

    truncate long docs, sliding window of 128 tokens (stride 64)

    tokenize using GPT-2 tokenizer

    split and save:

        data/train.bin → 90%

        data/val.bin → 10%

3. add_memory.py
modify model to support:

    STM: sliding attention window, context length = 128

    LTM: external content-addressable memory, max 256 items

routing mechanism:

    calculate token-level surprisal after softmax

    if surprisal > 4.5 bits:

        store token into LTM (FIFO if full)

    retrieve from LTM using cosine sim > 0.8

note:

    LTM is disabled by default unless --with-ltm is passed

    all memory is cleared between training sequences (no bleed)

4. train.py
args:

    --with-ltm: enable LTM routing + retrieval

    --baseline: disable LTM, use STM-only transformer (default)

config:

    data: data/train.bin

    optimizer: AdamW (lr=5e-4, betas=0.9/0.95, wd=0.1)

    scheduler: cosine w/ warmup (500 steps)

    steps: 10,000

    batch size: 8

    logging: tqdm + wandb

logging extras:

    LTM_enabled: true/false

    routed_token_ratio

    avg_surprisal_routed

    ltm_hit_rate

    tok/sec, loss, ETA

saves to:

    trained/with_ltm/ or trained/baseline/

5. eval.py
input:

    trained/with_ltm/

    trained/baseline/

    data/val.bin

metrics:

    bits per character (BPC)

    long-range dependency recall (custom synthetic test)

    memory hit rate (if LTM enabled)

    token/sec throughput

output (json):

{
  "baseline": {
    "bpc": ...,
    "long_range_recall": ...,
    "memory_hit_rate": 0.0,
    "tokens_per_sec": ...
  },
  "with_ltm": {
    "bpc": ...,
    "long_range_recall": ...,
    "memory_hit_rate": ...,
    "tokens_per_sec": ...
  }
}

save to:

    results.json

6. plot_eval.py
input:

    results.json

output:

    bar chart:

        BPC

        long-range recall

        tokens/sec

    line plot (optional):

        LTM hit rate

save:

    results.png

7. run_all.sh

#!/bin/bash
python download_models.py
python prepare_data.py
python add_memory.py
python train.py --baseline
python train.py --with-ltm
python eval.py
python plot_eval.py

this pipeline isolates memory as the variable. if LTM helps, you'll see BPC ↓ and recall ↑. if it doesn't, the whole memory-core hype is vibes.