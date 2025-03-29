import torch
import numpy as np
import matplotlib.pyplot as plt
import string

# NLP Generally uses a bracket like notation to denote special characters
START_TOKEN = "<S>"
STOP_TOKEN = "<E>"

def get_word_list():
    name_list = []
    with open("/app/3_makemore/names.txt", "r") as names:
        name_list = names.read().splitlines()

    return name_list

def get_word_stats(name_list:list[str]):
    count = 0
    longest = 0
    shortest = 10000
    all_chars = 0
    char_count = {}
    for word in name_list:
        count += 1
        longest = len(word) if len(word) > longest else longest
        shortest = len(word) if len(word) < shortest else shortest
        all_chars += len(word)
        for char in word: 
            if char_count.get(char) is None:
                char_count[char] = 0
            char_count[char] += 1

    char_counts = [(char, count) for char, count in char_count.items()]
    char_counts.sort(key=lambda x:x[1])

    return {
        "count": len(name_list),
        "longest": max(len(name) for name in name_list),
        "shortest": min(len(name) for name in name_list),
        "char_count": all_chars,
        "most_common_char": char_counts[-1],
        "least_common_char": char_counts[0],
        "char_counts": char_counts
    }

def get_bigram(name:str):
    padded = [START_TOKEN] + list(name) + [STOP_TOKEN]
    bigrams = []
    for char1, char2 in zip(padded, padded[1:]):
        bigrams.append((char1, char2))

    return bigrams

def get_bigrams(name_list:list[str]) -> dict[tuple[str,str], int]:
    """Returns a dict of unique bigrams and their frequency"""
    bigrams = {}
    for name in name_list:
        for bigram in get_bigram(name):
            bigrams[bigram] = bigrams.get(bigram, 0) + 1

    return bigrams

def get_char_index_map():
    chars = [START_TOKEN] + list(string.ascii_lowercase) + [STOP_TOKEN]

    return {
        char: index  
        for index, char
        in enumerate(chars)
    }

def plot_bigram_frequency(name_list:list[str]):
    # Define row and column labels
    char_index_map = get_char_index_map()
    labels = list(char_index_map.keys())
    size = len(labels)

    data = np.zeros((size, size))
    for bigram, count in get_bigrams(name_list).items():
        char1, char2 = bigram
        char1_index = char_index_map[char1]
        char2_index = char_index_map[char2]
        data[char1_index, char2_index] = count

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(16,16))

    # Plot heatmap using pcolormesh (shades of blue)
    cmap = plt.cm.Blues  # Colormap
    heatmap = ax.pcolormesh(data, cmap=cmap, shading='auto')

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Count", fontsize=12)

    # Annotate each cell with (row_char, col_char) and count
    for i in range(size):
        for j in range(size):
            ax.text(j + 0.5, i + 0.5, f"{labels[i]}{labels[j]}\n{data[i, j]}", 
                    ha='center', va='center', fontsize=8, color='black')

    # Set axis ticks and labels
    ax.set_xticks(np.arange(size) + 0.5)
    ax.set_yticks(np.arange(size) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Set labels and title
    ax.set_xlabel("Column Character", fontsize=12)
    ax.set_ylabel("Row Character", fontsize=12)
    ax.set_title("Character Pair Frequency Heatmap", fontsize=14)

    # Save as PNG
    plt.savefig("/app/3_makemore/figures/name_bigram_frequency.png", dpi=300, bbox_inches="tight")
    plt.show()

def bigram_tensor(name_list:list[str]):
    """Create a 28 x 28 tensor  that contains the pairwise bigram frequency.
    Note that we have a 28x28 matrix because we are store letters of the
    english alphabet (26) plus our two extra start and end characters"""
    char_index_map = get_char_index_map()

    tensor = torch.zeros(28, 28, dtype=torch.int32)
    for bigram, count in get_bigrams(name_list).items():
        char1, char2 = bigram
        char1_index = char_index_map[char1]
        char2_index = char_index_map[char2]
        tensor[char1_index, char2_index] = count
    
    return tensor

name_list = get_word_list()
tensor = bigram_tensor(name_list)
plot_bigram_frequency(name_list)

