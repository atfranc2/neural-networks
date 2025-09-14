import torch
import torch.nn.functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import string
import math
import random
import json
from matplotlib.animation import FuncAnimation

class NeuralModel:
    """
    Implements the Neural Probabilistic model described in: 
    Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. Journal of Machine Learning Research, 3(February), 1137-1155.
    """
    def __init__(self):
        self.START_TOKEN = "<S>"
        self.STOP_TOKEN = "<E>"

        self.source = "/app/3_makemore_v2/names.txt"
        self.__data = None

        self.is_split = False
        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []

        self.loss = None
        self.trained_weights = None
        
        self.valid_chars = [self.START_TOKEN] + list(string.ascii_lowercase) + [self.STOP_TOKEN]
        self.n_chars = len(self.valid_chars)
        self.generator = torch.Generator().manual_seed(824924294)

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
    def words(self):
        if self.__data is not None:
            return self.__data
        
        with open(self.source, "r") as names:
            self.__data = [name.lower() for name in names.read().splitlines()]

        random.seed(42)
        random.shuffle(self.__data)

        return self.__data
    
    def split(self, words:list[str], train=0.8, validate=0.1, test=0.1):
        # Assumes the original dataset has been shuffled
        N = len(words)
        train_stop = math.floor(N*train)
        train_data = words[:train_stop]
        validate_stop = math.floor(N*(train+validate))
        validate_data = words[train_stop:validate_stop]
        test_data = words[validate_stop:]

        return train_data, validate_data, test_data
    
    def n_gram_tensors(self, words:list[str], context_size=3):
        """
        Builds up the training set. 
        X contains the indexes of characters with len = {context_size} for observed words. 
        Y contains the next character give the previous {context_size} of characters. 
        So given context_size=3 we would have:
        X = [
            [char_index_1, char_index_2, char_index_3] # Word 1
            [char_index_1, char_index_2, char_index_3] # Word 1
            [char_index_1, char_index_2, char_index_3] # Word 1
            [char_index_1, char_index_2, char_index_3] # Word 2
            ...
            [char_index_1, char_index_2, char_index_3] # Word N
        ]
        Y = [
            char_index_4 # Word 1
            char_index_4 # Word 1
            char_index_4 # Word 1
            char_index_4 # Word 2
            ...
            char_index_4 # Word N
        ]
        """
        x, y = [], []
        start_index = self.char_index_map[self.START_TOKEN]
        for word in words:
            context = [start_index] * context_size
            for character in list(word) + [self.STOP_TOKEN]:
                character_index = self.char_index_map[character]
                x.append(context)
                y.append(character_index)
                # print("".join([self.index_char_map[index] for index in context]), "-->", character)
                context = context[1:] + [character_index] # Slide context window right across the word

        return torch.tensor(x), torch.tensor(y)

    def tahn_kaiming(self, fan_in:int) -> float:
        tanh_gain = 5/3
        return tanh_gain * (2 / fan_in)**0.5
    
    def train(self, epochs:int = 10000):
        # MLP Reference
        # BatchNorm Reference - https://arxiv.org/abs/1502.03167
        
        learning_rate = 0.8
        batch_size = 60
        hidden_size = 300
        embedding_size = 10
        self.context_size = 3

        self.train_words, self.validate_words, self.test_words = self.split(self.words)
        
        # Each x is a vector of length context size. Each element containing the character index
        self.X, self.Y = self.n_gram_tensors(self.train_words, self.context_size)
        
        # The embedding matrix. Vector representation of words, their relationships, and their context.
        self.C = torch.randn(self.n_chars, embedding_size) # 28 x 10

        # Applies Kaiming Init - https://arxiv.org/pdf/1502.01852
        # From PyTorch Reference - https://docs.pytorch.org/docs/stable/nn.init.html
        self.W1 = torch.randn(self.context_size * embedding_size, hidden_size) * self.tahn_kaiming(hidden_size)
        # Excluding bias on hidden layer because it is cancelled out by batch normalization

        self.W2 = torch.randn(hidden_size, self.n_chars) * 0.01
        self.b2 = torch.randn(1, self.n_chars) * 0.0

        # The initial normalization starts at a gaussian with mean=0 and std=1
        # However this may not be the optimal configuration. Therefore, the network
        # can adjust the spread of the distribution and where it is centered. 
        # We start with no scaling and no shifting (start gaussian)
        self.bn_scale = torch.ones(1, hidden_size) # Beta from paper
        self.bn_shift = torch.zeros(1, hidden_size) # Gamma from paper

        parameters = [self.C, self.W1, self.W2, self.b2, self.bn_scale, self.bn_shift]
        for parameter in parameters:
            parameter.requires_grad = True

        self.bn_running_mean = torch.zeros(1, hidden_size)
        self.bn_running_std = torch.ones(1, hidden_size)

        # Collect snapshots for animation
        self._snapshots = []

        for epoch in range(epochs):
            if epoch % 500 == 0:
                print(f"Epoch {epoch} / {epochs}")

            batch_indexes = torch.randint(0, self.X.shape[0], (batch_size,))
            training_examples = self.X[batch_indexes]

            # A three dimensional matrix. First dim: examples, Second dim: word embedding vectors, Third Dim: Embeddings 
            # Size: batch_size, context_size, embedding_size
            batch = self.C[training_examples]

            pre_activation = batch.view(batch_size, -1) @ self.W1
            # Sum down the columns (vertically)
            # Takes mean + std of all pre-activations across batch
            batch_mean = pre_activation.mean(0, keepdim=True) # Sum down the columns (vertically)
            batch_std = pre_activation.std(0, keepdim=True)

            with torch.no_grad():
                self.bn_running_mean = 0.999 * self.bn_running_mean + 0.001 * batch_mean
                self.bn_running_std = 0.999 * self.bn_running_std + 0.001 * batch_std

            bn_pre_activation = self.bn_scale * ((pre_activation - batch_mean) / batch_std) + self.bn_shift
            activation = torch.tanh(bn_pre_activation)

            logits = activation @ self.W2 + self.b2
            loss = F.cross_entropy(logits, self.Y[batch_indexes])

            for parameter in parameters: 
                parameter.grad = None

            # Capture snapshot every 500 iterations (pre-activation distribution + BN params)
            if epoch % 500 == 0:
                with torch.no_grad():
                    self._snapshots.append({
                        'epoch': epoch,
                        'pre_activation': pre_activation.detach().cpu().flatten().numpy(),
                        'bn_scale_mean': self.bn_scale.mean().item(),
                        'bn_shift_mean': self.bn_shift.mean().item(),
                        'bn_scale': self.bn_scale.detach().cpu().flatten().numpy(),
                        'bn_shift': self.bn_shift.detach().cpu().flatten().numpy(),
                    })

            loss.backward()
            
            for parameter in parameters:
                # print(parameter.grad)
                parameter.data += parameter.grad * -learning_rate

        self.loss = loss.data

        # Plotting/animation is handled outside the class after training
        
    
    def eval(self, split="validate"):
        data = self.train_words
        if split == "validate":
            data = self.validate_words
        elif split == "test":
            data = self.test_words
        
        X, Y = self.n_gram_tensors(data, self.context_size)
        input = self.C[X]
        dim_1, dim_2, dim_3 = input.shape
        pre_activation = input.view(dim_1, dim_2*dim_3) @ self.W1
        norm_pre_activation = self.bn_scale * ((pre_activation - self.bn_running_mean) / self.bn_running_std) + self.bn_shift
        activation = torch.tanh(norm_pre_activation)

        logits = activation @ self.W2 + self.b2
        mean_nll = F.cross_entropy(logits, Y)
        print("Train Loss:", self.loss, "Train Loss:", mean_nll.item())
        return mean_nll
    

model = NeuralModel()
model.train(epochs=10000)

# Build animation outside the class: only pre-activation histogram
if len(model._snapshots) > 0:
    # Compute global x-limits across all snapshots and fixed bins
    all_vals = np.concatenate([s['pre_activation'] for s in model._snapshots])
    xmin, xmax = float(all_vals.min()), float(all_vals.max())
    # Add small padding so extremes don't clip
    pad = 0.05 * (xmax - xmin + 1e-8)
    xmin -= pad
    xmax += pad
    bins = np.linspace(xmin, xmax, 50)

    fig, ax = plt.subplots(figsize=(8, 4))

    def update_hist(i):
        snap = model._snapshots[i]
        ax.cla()
        ax.hist(snap['pre_activation'], bins=bins, color='tab:blue', alpha=0.75)
        # Means of bn_scale and bn_shift formatted to 4 decimals
        gamma_mean = float(np.mean(snap['bn_scale_mean']))
        beta_mean = float(np.mean(snap['bn_shift_mean']))
        title = f"Pre-activations — epoch {snap['epoch']} | gamma={gamma_mean:.4f}, beta={beta_mean:.4f}"
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.set_xlim(xmin, xmax)
        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, update_hist, frames=len(model._snapshots), interval=1200, blit=False, repeat_delay=1500)
    try:
        from matplotlib.animation import PillowWriter
        anim.save('3_makemore_v3_batchnorm/preactivations.gif', writer=PillowWriter(fps=1))
        print("Saved animation to 3_makemore_v3_batchnorm/preactivations.gif")
    except Exception as e:
        print("Could not save GIF animation:", e)

print(f"Train Loss: {model.loss}")
model.eval("test")
