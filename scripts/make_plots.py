import matplotlib.pyplot as plt

def plot_compression_results(vocab_sizes, compression_ratios):
    """
    Generates a plot of the compression ratio against vocabulary size.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(vocab_sizes, compression_ratios, marker='o', linestyle='-', color='#0055a4')
    
    plt.title("Compression Ratio vs. Vocabulary Size")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("Bytes per Token")
    
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xticks(vocab_sizes)
    
    # Save the plot for the report
    plt.tight_layout()
    plt.savefig("compression_ratio_study.png", dpi=300)
    plt.show()