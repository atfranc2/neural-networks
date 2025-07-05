import torch
import torch.nn.functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt
import string
import math
import random

# NLP Generally uses a bracket like notation to denote special characters

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
    fig, ax = plt.subplots(figsize=(30,30))

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

# name_list = get_word_list()
# tensor = bigram_tensor(name_list)
# plot_bigram_frequency(name_list)

def tensor_to_csv(tensor:torch.Tensor, file_name:str|None=None):
    file_name = file_name if file_name is not None else "bigram_frequncy.csv"
    with open(f"/app/3_makemore/data/{file_name}", "w") as file:
        writer = csv.writer(file)

        for row_index in range(tensor.shape[0]):
            writer.writerow([value.item() for value in tensor[row_index]])

def csv_to_tensor(file_name, transform_fn):
    data = []
    with open(f"/app/3_makemore/data/{file_name}", "r") as file:
        reader = csv.reader(file)
        for line in reader:
            data.append([transform_fn(value) for value in line])

    return torch.tensor(data)

# print(csv_to_tensor("bigram_frequncy.csv", lambda value: int(value)))

class BigramModel:
    """A naive straight probability, table based approach."""
    def __init__(self):
        self.START_TOKEN = "<S>"
        self.STOP_TOKEN = "<E>"

        self.source = "/app/3_makemore/names.txt"
        self.__data = None

        self.valid_chars = [self.START_TOKEN] + list(string.ascii_lowercase) + [self.STOP_TOKEN]
        self.__char_index_map = None
        self.__index_char_map = None
        self.__bigram_count_map = None
        self.__bigram_frequency_matrix = None
        self.__bigram_probability_matrix = None

    @property
    def char_index_map(self):
        if self.__char_index_map is not None: 
            return self.__char_index_map
        
        self.__char_index_map = {
            char:index
            for index, char
            in enumerate(self.valid_chars)
        }

        return self.__char_index_map

    @property
    def index_char_map(self):
        if self.__index_char_map is not None: 
            return self.__index_char_map
        
        self.__index_char_map = {
            index: char
            for char, index
            in self.__char_index_map.items()
        }

        return self.__index_char_map

    @property
    def data(self):
        if self.__data is not None:
            return self.__data
        
        with open(self.source, "r") as names:
            self.__data = [name.lower() for name in names.read().splitlines()]

        return self.__data

    @property
    def bigram_count_map(self):
        if self.__bigram_count_map is not None:
            return self.__bigram_count_map
        
        self.__bigram_count_map = {}
        for word in self.data:
            for bigram in self.get_bigrams(word):
                self.__bigram_count_map[bigram] = self.__bigram_count_map.get(bigram, 0) + 1
        return self.__bigram_count_map
    
    @property
    def bigram_frequency_matrix(self):
        if self.__bigram_frequency_matrix is not None:
            return self.__bigram_frequency_matrix
        
        num_chars = len(self.valid_chars)
        self.__bigram_frequency_matrix = torch.zeros((num_chars, num_chars))
        for char1 in self.valid_chars:
            char1_index = self.char_index_map.get(char1)
            for char2 in self.valid_chars:
                char2_index = self.char_index_map.get(char2)
                self.__bigram_frequency_matrix[char1_index, char2_index] = self.bigram_count_map.get((char1, char2), 0)
        
        return self.__bigram_frequency_matrix
    
    @property
    def bigram_probability_matrix(self):
        if self.__bigram_probability_matrix is not None:
            return self.__bigram_probability_matrix
        
        # The frequency of each bigram, divided by the sum of all frequencies in a row
        self.__bigram_probability_matrix = self.bigram_frequency_matrix / self.bigram_frequency_matrix.sum(dim=1, keepdim=True)

        return self.__bigram_probability_matrix
    
    def get_bigrams(self, word:str):
        padded_word = [self.START_TOKEN] + list(word) + [self.STOP_TOKEN]
        for char1, char2 in zip(padded_word, padded_word[1:]):
            yield (char1, char2)
    
    def is_stop_char(self, char:str):
        return char == self.STOP_TOKEN
    
    def get_char_index(self, char:str):
        return self.char_index_map[char]
    
    def get_char(self, index:int):
        return self.index_char_map[index]
    
    def get_bigram_prob(self, char1, char2):
        char1_index =self.get_char_index(char1)
        char2_index = self.get_char_index(char2)
        return self.bigram_probability_matrix[char1_index, char2_index]

    def make(self, n=1) -> list[str]:
        generator = torch.Generator().manual_seed(2147483647)
        names = []
        for index in range(n):
            char_row_index = 0 # Always begin in the row containing the start character
            name_chars = []
            while True:
                char_probs = self.bigram_probability_matrix[char_row_index]
                char_column_index = torch.multinomial(char_probs, num_samples=1, generator=generator).item()
                char = self.get_char(char_column_index)
                if self.is_stop_char(char):
                    break

                name_chars.append(char)
                char_row_index = char_column_index
            
            names.append("".join(name_chars))

        return names

    def eval(self, words:list[str]):
        """
        Calculate the negative log likelihood (NLL) against a set of words. Lower is better because as probability approaches 1 the NLL approaches 0.
        """
        log_likelihoods = []
        for word in words:
            for bigram in self.get_bigrams(word.lower()):
                prob = self.get_bigram_prob(*bigram)
                logprob = torch.log(prob)
                log_likelihoods.append(logprob)

        # print(f"""
        # Log-Likelihood: {log_likelihood:.4f}
        # Negative Log-Likelihood (lower is better): {-log_likelihood:.4f}
        # Average Negative Log-Likelihood: {(-log_likelihood/len(words)):.4f}
        # """)
        return -sum(log_likelihoods) / len(log_likelihoods)


model = BigramModel()
print("Naive Bigram Loss:", model.eval(model.data))
print("Naive Bigram:", model.make(10))


class NetBigramModel:
    """A network based appraoch."""
    def __init__(self):
        self.START_TOKEN = "<S>"
        self.STOP_TOKEN = "<E>"

        self.source = "/app/3_makemore/names.txt"
        self.__data = None

        self.is_split = False
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []

        self.loss = None
        self.trained_weights = None
        
        self.valid_chars = [self.START_TOKEN] + list(string.ascii_lowercase) + [self.STOP_TOKEN]
        self.n_chars = len(self.valid_chars)

        self.__char_index_map = None
        self.__index_char_map = None
        self.__bigram_count_map = None
        self.__bigram_frequency_matrix = None
        self.__bigram_probability_matrix = None

    @property
    def char_index_map(self):
        if self.__char_index_map is not None: 
            return self.__char_index_map
        
        self.__char_index_map = {
            char:index
            for index, char
            in enumerate(self.valid_chars)
        }

        return self.__char_index_map

    @property
    def index_char_map(self):
        if self.__index_char_map is not None: 
            return self.__index_char_map
        
        self.__index_char_map = {
            index: char
            for char, index
            in self.__char_index_map.items()
        }

        return self.__index_char_map

    @property
    def data(self):
        if self.__data is not None:
            return self.__data
        
        with open(self.source, "r") as names:
            self.__data = [name.lower() for name in names.read().splitlines()]

        return self.__data

    @property
    def bigram_count_map(self):
        if self.__bigram_count_map is not None:
            return self.__bigram_count_map
        
        self.__bigram_count_map = {}
        for word in self.data:
            for bigram in self.get_bigrams(word):
                self.__bigram_count_map[bigram] = self.__bigram_count_map.get(bigram, 0) + 1
        return self.__bigram_count_map
    
    @property
    def bigram_frequency_matrix(self):
        if self.__bigram_frequency_matrix is not None:
            return self.__bigram_frequency_matrix
        
        num_chars = len(self.valid_chars)
        self.__bigram_frequency_matrix = torch.zeros((num_chars, num_chars))
        for char1 in self.valid_chars:
            char1_index = self.char_index_map.get(char1)
            for char2 in self.valid_chars:
                char2_index = self.char_index_map.get(char2)
                self.__bigram_frequency_matrix[char1_index, char2_index] = self.bigram_count_map.get((char1, char2), 0)
        
        return self.__bigram_frequency_matrix
    
    @property
    def bigram_probability_matrix(self):
        if self.__bigram_probability_matrix is not None:
            return self.__bigram_probability_matrix
        
        # The frequency of each bigram, divided by the sum of all frequencies in a row
        self.__bigram_probability_matrix = self.bigram_frequency_matrix / self.bigram_frequency_matrix.sum(dim=1, keepdim=True)

        return self.__bigram_probability_matrix
    
    def get_bigrams(self, word:str):
        padded_word = [self.START_TOKEN] + list(word) + [self.STOP_TOKEN]
        for char1, char2 in zip(padded_word, padded_word[1:]):
            yield (char1, char2)
    
    def is_stop_char(self, char:str):
        return char == self.STOP_TOKEN
    
    def get_char_index(self, char:str):
        return self.char_index_map[char]
    
    def get_char(self, index:int):
        return self.index_char_map[index]
    
    def test_train_split(self, size = None, train_prop=0.8):
        if self.is_split:
            return self.x_train, self.y_train, self.x_test, self.y_test

        size = size if size is not None else len(self.data)
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []
        train_size = math.floor(size * train_prop)

        index_range = np.random.default_rng(213242545)
        sample_indexes = np.arange(0, size)
        index_range.shuffle(sample_indexes)

        curr_train_size = 0
        for sample_index in sample_indexes:
            word = self.data[sample_index]
            if curr_train_size < train_size:
                for char1, char2 in self.get_bigrams(word):
                    char1_index = self.get_char_index(char1)
                    char2_index = self.get_char_index(char2)
                    self.x_train.append(char1_index)
                    self.y_train.append(char2_index)
                curr_train_size += 1
            else: 
                for char1, char2 in self.get_bigrams(word):
                    char1_index = self.get_char_index(char1)
                    char2_index = self.get_char_index(char2)
                    self.x_test.append(char1_index)
                    self.y_test.append(char2_index)
        
        self.is_split = True

        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def encode(self, tensor:torch.Tensor):
        """One hot encode character index values to be one hot encoded vectors of equal length"""
        return F.one_hot(tensor, num_classes=len(self.valid_chars)).float()
    
    def train(self, learning_rate=10, epochs=10, regularization_strength=0.1):
        # Creates arrays of character indexes of the input and output characters
        x_train, y_train, x_test, y_test = self.test_train_split(train_prop=0.8)
        x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
        x_train_enc = self.encode(x_train)
        # Initialize the weights with random values from the standard normal distribution
        generator = torch.Generator().manual_seed(2147483647)
        weights = torch.randn((self.n_chars, self.n_chars), generator=generator, requires_grad=True)

        loss = torch.tensor(0)
        prev_loss = torch.tensor(0)
        for epoch in range(epochs):
            # print(f"epoch {epoch + 1} / {epochs}")
            # Initial output. Do matrix multiplication on inputs and weights ([n_examples x 28] x [28, 28] => [n_examples x 28])
            log_counts = x_train_enc @ weights

            # Transform the input so that is cannot be negative. Will give "count" like information. This is then normalized. Also call logits. This is done using the softmax function.
            # This is done with pairwise exponentiation, followed by normalization by row sum. This is the same that was done for calculating bigram probabilities in the table-based approach. 
            counts = log_counts.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            
            # Givens an example at index x, find the probability at index y. Calculate the negative mean log likelihood
            regularization = regularization_strength * (weights**2).mean() # Makes output probabilities more uniform. Helps to overcome non-uniformities in data such as 0 counts for some character pairs. 
            loss = -1 * probs[torch.arange(len(x_train)), y_train].log().mean() + regularization
            weights.grad = None # More efficient than setting to 0
            loss.backward()
            weights.data += -learning_rate * weights.grad
            # print(f"Loss: {loss} (delta={loss - prev_loss})")
            prev_loss = loss

        self.loss = loss
        self.trained_weights = weights

    def eval(self):
        if self.x_test is None or len(self.x_test) == 0:
            return -1
        
        x_test = torch.tensor(self.x_test)
        xenc = self.encode(x_test)
        log_counts = xenc @ self.trained_weights
        counts = log_counts.exp()
        probs = counts / counts.sum(1, keepdim=True)

        loss = probs[torch.arange(len(self.x_test)), self.y_test].log().mean() * -1
        print("Training Loss:", self.loss)
        print("Out of Sample Loss:", loss)

    def predict(self, N=5):
        if self.trained_weights is None:
            raise Exception("Model has not been trained")

        generator = torch.Generator().manual_seed(2147483647)
        predictions = []
        for n in range(N):
            xenc = F.one_hot(torch.tensor([self.get_char_index(self.START_TOKEN)]), num_classes=len(self.valid_chars)).float()
            prediction = ""
            while True:
                log_counts = xenc @ self.trained_weights
                counts = log_counts.exp()
                char_probs = counts / counts.sum(dim=1, keepdim=True)
                char_column_index = torch.multinomial(char_probs, num_samples=1, generator=generator).item()

                if self.get_char(char_column_index) == self.STOP_TOKEN:
                    predictions.append(prediction)
                    break

                xenc = F.one_hot(torch.tensor([char_column_index]), num_classes=len(self.valid_chars)).float()
                prediction += self.get_char(char_column_index)

        return predictions


net_model = NetBigramModel()
net_model.train(epochs=100, learning_rate=50, regularization_strength=0.01)
print("Network Bigram Loss:", net_model.loss)
print("Network Bigram:", net_model.predict(N=10))
print("Evaluation:", net_model.eval())
