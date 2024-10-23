import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_topic_weights(topic_weights, top_words, num_topics=5, num_words=10):
    """
    Create a simple bubble plot of topic weights.
    
    Parameters:
    -----------
    topic_weights : pandas.DataFrame
        Output from normalize_and_visualize_topic_weights (normalized weights)
    top_words : pandas.DataFrame
        Output from normalize_and_visualize_topic_weights (top words per topic)
    num_topics : int
        Number of topics to visualize
    num_words : int
        Number of top words to show per topic
    """
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # For each topic
    for topic_idx in range(num_topics):
        # Get top words for this topic
        words = top_words[f"Topic #{topic_idx}"].head(num_words)
        
        # Get weights for these words
        weights = [topic_weights.loc[word, f"Topic #{topic_idx}"] for word in words]
        
        # Create bubbles
        plt.scatter([topic_idx] * num_words,  # x-coordinates (same for all words in topic)
                   range(num_words),          # y-coordinates (0 to num_words)
                   s=[w * 2000 for w in weights],  # bubble sizes
                   alpha=0.6,
                   label=f"Topic {topic_idx}")
        
        # Add word labels
        for i, word in enumerate(words):
            plt.annotate(word, 
                        (topic_idx, i),
                        xytext=(10, 0), 
                        textcoords='offset points')
    
    # Customize plot
    plt.title("Topic Words Distribution")
    plt.xlabel("Topics")
    plt.ylabel("Words")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(-1, num_words)
    
    plt.tight_layout()
    plt.show()