import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Try to load results_v2.json (which includes all three models), else fallback to results.json
    try:
        with open("results_v2.json") as f:
            results = json.load(f)
        models = ['baseline', 'with_ltm', 'with_ltm_v2']
        colors = ['blue', 'orange', 'green']
    except FileNotFoundError:
        with open("results.json") as f:
            results = json.load(f)
        models = ['baseline', 'with_ltm']
        colors = ['blue', 'orange']
    
    # Extract metrics (handle missing models gracefully)
    present_models = [m for m in models if m in results]
    metrics = {
        'bpc': [results[m]['bpc'] for m in present_models],
        'tokens_per_sec': [results[m]['tokens_per_sec'] for m in present_models]
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot BPC
    ax1.bar(present_models, metrics['bpc'], color=colors[:len(present_models)])
    ax1.set_title('Bits per Character (BPC)')
    ax1.set_ylabel('BPC (lower is better)')
    
    # Plot Tokens/sec
    ax2.bar(present_models, metrics['tokens_per_sec'], color=colors[:len(present_models)])
    ax2.set_title('Inference Speed')
    ax2.set_ylabel('Tokens per Second (higher is better)')
    
    # Add hit rate plot if available (for with_ltm or with_ltm_v2)
    for m, c in zip(['with_ltm', 'with_ltm_v2'], ['orange', 'green']):
        if m in results and 'memory_hit_rate' in results[m]:
            fig2, ax4 = plt.subplots(figsize=(5, 5))
            ax4.plot(results[m]['memory_hit_rate'], color=c, label=m)
            ax4.set_title(f'{m} LTM Hit Rate Over Time')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Hit Rate')
            ax4.legend()
            fig2.tight_layout()
            fig2.savefig(f'hit_rate_{m}.png', dpi=300)
    
    # Adjust layout and save
    fig.suptitle('RevenaHybrid Evaluation Results', y=1.05)
    fig.tight_layout()
    fig.savefig('results_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()