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

def plot_loss_compare(model1:"NeuralModel", model2:"NeuralModel"):
    # Create figure and two subplots side by side
    fig, (rand_start_loss_ax, adjust_start_loss_ax) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    rand_start_loss_ax:plt.Axes
    adjust_start_loss_ax:plt.Axes
    # First line chart

    rand_start_loss_ax.plot(range(1, len(model1.losses)+1), model1.losses, label='Random', color='blue')
    rand_start_loss_ax.set_xlim(-10)
    rand_start_loss_ax.set_ylim(0, 35)
    rand_start_loss_ax.set_title('Random Starting Position')
    rand_start_loss_ax.set_xlabel('Training Epoch')
    rand_start_loss_ax.set_ylabel('Negative Mean Log Likelihood')
    rand_start_loss_ax.grid(True)
    rand_start_loss_ax.legend()

    adjust_start_loss_ax.plot(range(1, len(model2.losses)+1), model2.losses, label='Adjusted', color='red')
    adjust_start_loss_ax.set_xlim(-10)
    adjust_start_loss_ax.set_ylim(0, 35)
    adjust_start_loss_ax.set_title('Adjusted Starting Position')
    adjust_start_loss_ax.set_xlabel('Training Epoch')
    adjust_start_loss_ax.set_ylabel('Negative Mean Log Likelihood')
    adjust_start_loss_ax.grid(True)
    adjust_start_loss_ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig("/app/3_makemore_v2/figures/random_vs_optimized_start_loss.png", dpi=300, bbox_inches="tight")

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
    
    def get_C(self, embedding_size=2):
        return torch.randn(self.n_chars, embedding_size, generator=self.generator, requires_grad=True)
    
    def forwards(self, data:torch.Tensor):
        o1 = (data @ self.W1) + self.b1
        o1_activated = torch.tanh(o1)
        return (o1_activated @ self.W2) + self.b2
    
    def model(self, char_indexes:list[int]):
        dim_1, dim_2 = self.C[char_indexes].shape
        logits = self.forwards(self.C[char_indexes].view(1, dim_1*dim_2))
        return F.softmax(logits, dim=1)
    
    def pickle(self):
        with open("/app/3_makemore_v2/pickles/C.json", "w") as C:
            json.dump(self.C.tolist(), C)

        with open("/app/3_makemore_v2/pickles/W1.json", "w") as W1:
            json.dump(self.W1.tolist(), W1)

        with open("/app/3_makemore_v2/pickles/b1.json", "w") as b1:
            json.dump(self.b1.tolist(), b1)

        with open("/app/3_makemore_v2/pickles/W2.json", "w") as W2:
            json.dump(self.W2.tolist(), W2)

        with open("/app/3_makemore_v2/pickles/b2.json", "w") as b2:
            json.dump(self.b2.tolist(), b2)

        with open("/app/3_makemore_v2/pickles/params.json", "w") as params:
            json.dump({
                "train_words": self.train_words, 
                "validate_words": self.validate_words, 
                "test_words": self.test_words,
                "loss": self.loss,
                "context_size": self.context_size,
                "embedding_size": self.embedding_size
            }, params)

    def load(self):
        with open("/app/3_makemore_v2/pickles/C.json", "r") as C:
            self.C = torch.tensor(json.load(C))

        with open("/app/3_makemore_v2/pickles/W1.json", "r") as W1:
            self.W1 = torch.tensor(json.load(W1))

        with open("/app/3_makemore_v2/pickles/b1.json", "r") as b1:
            self.b1 = torch.tensor(json.load(b1))

        with open("/app/3_makemore_v2/pickles/W2.json", "r") as W2:
            self.W2 = torch.tensor(json.load(W2))

        with open("/app/3_makemore_v2/pickles/b2.json", "r") as b2:
            self.b2 = torch.tensor(json.load(b2))

        with open("/app/3_makemore_v2/pickles/params.json", "r") as params:
            jparams = json.load(params)
            self.train_words = jparams["train_words"]
            self.validate_words = jparams["validate_words"]
            self.test_words = jparams["test_words"]
            self.loss = jparams["loss"]
            self.context_size = jparams["context_size"]
            self.embedding_size = jparams["embedding_size"]

    def train(
        self, 
        context_size=3, 
        embedding_size=2, 
        hidden_size=10, 
        epochs=10,
        regularization_strength=None,
        learning_rate=0.1,
        learning_rate_decay=True,
        batch_size=None,
        pickle=True,
        weight_adjustments=None,
    ):
        self.train_words, self.validate_words, self.test_words = self.split(self.words)
        # self.train_words, self.validate_words, self.test_words = self.words,self.words,self.words
        X, Y = self.n_gram_tensors(self.train_words, context_size)

        # This will hold the vector embeddings of each of the characters
        self.C = self.get_C(embedding_size)
        
        """
        The input layer size will be a vector with length {context_size} x {embedding_size}. 
        Each prediction uses {context_size} characters to do next character prediction. And each character is represented by a {embedding_size} vector.
        The hidden layer is a hyperparameter and is of variable length. 
        The output layer is of size {n_valid_chars}
        
        Example: 
        context_size = 3
        embedding_size = 2

        I.shape = (1,6) 
        W1.shape = (6, hidden_size)
        b1.shape = (1, hidden_size)

        W2.shape = (hidden_size, n_valid_chars)
        b2.shape = (1, n_valid_chars)

        """
        
        # Each input character vector must be a (1, {context_size} x {embedding_size}) vector. C[X] gives ({context_size}, {embedding_size}) vectors. 
        # Must reshape the input to make it amenable to matrix multiplication
        
        # Now we implement the network architecture described in Bengio et. al.
        """
        When initializing the state of our network we want to make it such that each 
        character has an equal probability of being selected on average. Why? This gives
        us the lowest possible starting loss. How do we measure success? When all 28 of the 
        characters have an equal probabilities (1/28=0.035=3.5%). For this to occur the logits 
        being generated by the network all have to be roughly equal and/or near zero. 

        Essentially what this means is that at initialization time we are attributing an arbitrarily
        high confidence to incorrect output in the network. This lead to the loss being inflated. This 
        leads to what is essentially wasted computation.

        When this condition is met all the values have roughly equal probabilities and we would
        expect the loss at initialization to be: init_loss = -log(1/28) = 3.33. So 3.33 is the 
        target for the initial loss of the network. 

        We can achieve those by setting the bias of the logit layer to 0 and scaling the weights
        down towards 0. And this works quite well. Before implementing this some of the initial 
        loss values were about 38.0. That is very high after zeroing they were down near about 5.0.

        So why not just initialize the biases and the weights in the logit layer?

        The other issue that must be dealt with at the hidden activations. Many of these values
        are large (greater than about 1.5 - 2). This matters because we are using tanh as the activation 
        function because it means the output is -1 or 1. Then the derivative of tanh = 1-tanh()**2, which 
        when most of your values come out to 0. But we use the derivative to calculate gradients! So many
        of the gradients come out to 0 which mean we stop being able to nudge the network towards the optimal 
        state. 

        When all examples during training push the activation into a tail it basically creates a "dead neuron"
        that cannot learn. A high gradient occurring by change can nudge a weight towards such extreme values 
        in some iterations that the neuron can never learn again (example used for RELU)


        More tanh facts:
        - It only ever decreases the gradient passing through it. When it is at an extreme (-1 or 1) it squashes the gradient.
        When it is at its middle (0) it does not modify the gradient at all because it equals 1. 

        """
        
        weight_adjustments = weight_adjustments if weight_adjustments else []
        g1 = torch.Generator().manual_seed(3148245075)
        self.W1 = torch.randn(context_size*embedding_size, hidden_size, generator=g1) * self.safe_pop(weight_adjustments, 0, 1)
        
        g2 = torch.Generator().manual_seed(3143145075)
        self.b1 = torch.randn(1, hidden_size, generator=g2) * self.safe_pop(weight_adjustments, 0, 1)

        # Multiplying by 0.1 helps bring our initial logits down towards zero at initialization
        g3 = torch.Generator().manual_seed(31958345075)
        self.W2 = torch.randn(hidden_size, self.n_chars, generator=g3) * self.safe_pop(weight_adjustments, 0, 1)
        
        # What initial logits to be near zero, so to start we add no bias
        g4 = torch.Generator().manual_seed(59535282)
        self.b2 = torch.randn(1, self.n_chars, generator=g4) * self.safe_pop(weight_adjustments, 0, 1)

        parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        print(f"Parameter Count: {sum([p.nelement() for p in parameters])}")
        for parameter in parameters:
            parameter.requires_grad = True

        self.losses = []
        self.pre_activations = []
        self.activations = []
        for epoch in range(epochs):
            if epoch % 5000 == 0: 
                print(f"Epoch {epoch}/{epochs}")
            
            batch_indexes = torch.randint(0, X.shape[0], (batch_size,)) if batch_size else None
            train_X = self.C[X[batch_indexes]] if batch_size else self.C[X]
            train_Y = Y[batch_indexes] if batch_size else Y
            dim_1, dim_2, dim_3 = train_X.shape
            # logits = self.forwards(train_X.view(dim_1, dim_2*dim_3))

            o1 = (train_X.view(dim_1, dim_2*dim_3) @ self.W1) + self.b1
            o1_activated = torch.tanh(o1)

            # Makes certain operations more efficient
            with torch.no_grad():
                self.pre_activations.extend(o1.tolist())
                self.activations.extend(o1_activated.tolist())
            
            logits =  (o1_activated @ self.W2) + self.b2

            # Calculate the negative mean log-likelihood
            # Forward pass done in batch, so probs will be a matrix of probability vectors equal to the number of rows in X. 
            # For each of this vectors we want to pull out the probability stored at the associated index in Y
            l2_norm = torch.tensor(0)
            if regularization_strength:
                l2_norm = sum([parameter.pow(2.0).sum() for parameter in parameters]) * regularization_strength

            mean_nll = F.cross_entropy(logits, train_Y) + l2_norm # Handles edge cases and is more efficient for forward and back pass. 
            self.losses.append(mean_nll.item())

            for parameter in parameters:
                parameter.grad = None

            mean_nll.backward()

            # Reduce learning rate by 10% in the last 10% of training epochs (learning rate decay)
            decayed_lr = learning_rate
            if learning_rate_decay: 
                decayed_lr = learning_rate * 0.1 if epoch > epochs * 0.9 else learning_rate

            for parameter in parameters:
                parameter.data += parameter.grad * -decayed_lr

        # print([loss.item() for loss in losses])
        # print(mean_nll.item())
        
        self.loss = mean_nll.item()
        self.context_size = context_size
        self.embedding_size = embedding_size

        if pickle:
            self.pickle()

        return self.losses[-1]
    
    def eval(self, split="validate"):
        data = self.train_words
        if split == "validate":
            data = self.validate_words
        elif split == "test":
            data = self.test_words
        
        X, Y = self.n_gram_tensors(data, self.context_size)
        dim_1, dim_2, dim_3 = self.C[X].shape
        logits = self.forwards(self.C[X].view(dim_1, dim_2*dim_3))
        mean_nll = F.cross_entropy(logits, Y)
        print("Train Loss:", self.loss, "Train Loss:", mean_nll.item())
        return mean_nll
    
    def predict(self, n=10):
        start_index = self.char_index_map[self.START_TOKEN]
        stop_index = self.char_index_map[self.STOP_TOKEN]
        for name_index in range(n):
            input = [start_index] * self.context_size
            prediction = ''
            while True: 
                probs = self.model(input)
                pred_index = torch.multinomial(probs, num_samples=1, replacement=False, generator=self.generator)
                if pred_index.item() == stop_index:
                    break

                prediction += self.index_char_map[pred_index.item()]
                input = input[1:] + [pred_index.item()]
                # print("".join([self.index_char_map[i] for i in input]))
            
            print(prediction)
        
    def tune_learning_rate(
            self, 
            steps=100,
            context_size=3, 
            embedding_size=2, 
            hidden_size=10, 
            batch_size=None
    ):
        train_words, _, _ = self.split(self.words)
        X, Y = self.n_gram_tensors(train_words, context_size)
        C = self.get_C(embedding_size)
        
        g1 = torch.Generator().manual_seed(3148245075)
        W1 = torch.randn(context_size*embedding_size, hidden_size, requires_grad=True, generator=g1)

        g2 = torch.Generator().manual_seed(3143145075)
        b1 = torch.randn(1, hidden_size, requires_grad=True, generator=g2)

        g3 = torch.Generator().manual_seed(31958345075)
        W2 = torch.randn(hidden_size, self.n_chars, requires_grad=True, generator=g3)
        
        g4 = torch.Generator().manual_seed(59535282)
        b2 = torch.randn(1, self.n_chars, requires_grad=True, generator=g4)

        parameters = [W1, b1, W2, b2]

        losses = []
        learning_rates_exp = torch.linspace(-3, 0, steps)
        learning_rates = 10**learning_rates_exp

        for learning_rate in learning_rates:
            batch_indexes = torch.randint(0, X.shape[0], (batch_size,)) if batch_size else None
            train_X = C[X[batch_indexes]]  if batch_size else C[X]
            train_Y = Y[batch_indexes] if batch_size else Y

            dim_1, dim_2, dim_3 = train_X.shape
            o1 = (train_X.view(dim_1, dim_2*dim_3) @ W1) + b1
            o1_activated = torch.tanh(o1)
            logits = o1_activated @ W2 + b2
            
            mean_nll = F.cross_entropy(logits, train_Y)
            losses.append(mean_nll.item())
            # print(mean_nll.item())

            for parameter in parameters:
                parameter.grad = None

            mean_nll.backward()

            for parameter in parameters:
                parameter.data += parameter.grad * -learning_rate

        return learning_rates_exp, losses

    def explore(
            self, 
            steps=100,
            context_size=3, 
    ):

        learning_rates_exp = torch.linspace(-3, 0, 10)
        learning_rates = [0.2, 0.4, 0.8]

        regularization_strengths_exp = torch.linspace(-6, 0, 10)
        regularization_strengths = [0.00001, 0.0001, 0.001]

        embedding_sizes = [2, 4, 6]
        hidden_sizes = [40, 100, 150]
        batch_sizes = [50]

        tests = []
        for learning_rate in learning_rates:
            for regularization_strength in regularization_strengths:
                for hidden_size in hidden_sizes:
                    for batch_size in batch_sizes:
                        for embedding_size in embedding_sizes: 
                            tests.append((learning_rate, batch_size, embedding_size, regularization_strength, hidden_size))

        loss_compares = []
        count = 1
        for learning_rate, batch_size, embedding_size, regularization_strength, hidden_size in tests:
            print(f"Running Test {count}/{len(tests)}")
            count += 1
            last_loss = model1.train(
                context_size=context_size, 
                embedding_size=embedding_size, 
                hidden_size=hidden_size, 
                epochs=5,
                regularization_strength=regularization_strength,
                learning_rate=learning_rate
            )
            loss_compares.append((learning_rate, batch_size, embedding_size, regularization_strength, hidden_size, last_loss))

        loss_compares.sort(key=lambda item: item[-1])
        print(loss_compares[:10])

        return loss_compares

    def plot_embeddings(self):
        C = self.C.detach().numpy()
        did_pca = False
        if self.embedding_size > 2:
            C_centered = C - C.mean(axis=0)
            # 2. Compute the covariance matrix
            cov = np.cov(C_centered, rowvar=False)  # shape: (5, 5)
            # 3. Get eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(cov)  # use eigh for symmetric matrices
            # 4. Sort eigenvectors by eigenvalues in descending order
            sorted_idx = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, sorted_idx]
            eigvals = eigvals[sorted_idx]

            # 5. Project data onto the first 2 principal components
            data = C_centered @ eigvecs[:, :2]  # shape: (100, 2)
            did_pca = True

        x, y, labels = [], [], []
        for index, (_x, _y) in enumerate(data):
            x.append(_x.item())
            y.append(_y.item())
            labels.append(model1.index_char_map[index])

        fig, ax = plt.subplots()

        # Create scatter plot with large markers
        plt.scatter(x, y, s=100, color='skyblue', edgecolors='black', zorder=2)

        # Add text labels at each point
        for xi, yi, label in zip(x, y, labels):
            plt.text(xi, yi, label, ha='center', va='center', fontsize=4, fontweight='bold', zorder=3)

        # Styling
        plt.title(f"Labeled Scatter Plot with Large Markers {'(PCA)' if did_pca else ''}")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("/app/3_makemore_v2/figures/embeddings.png", dpi=300, bbox_inches="tight")
        plt.show()
    
    def safe_pop(self, array:list, index=-1, default=None):
        try:
            return array.pop(index)
        except IndexError:
            return default

model1 = NeuralModel()
model2 = NeuralModel()
# model.explore()
# model.train(
#     context_size=3, 
#     embedding_size=10, 
#     hidden_size=500, 
#     epochs=30000,
#     regularization_strength=0.00001,
#     learning_rate=0.5,
#     batch_size=500,
#     pickle=False
# )
# model.load()
# model.eval("test")
# model.predict(n=100)
# model.plot_embeddings()
epochs = 1
model1.train(
    context_size=3, 
    embedding_size=10, 
    hidden_size=300, 
    epochs=epochs,
    regularization_strength=0.00001,
    learning_rate=0.5,
    batch_size=500,
    pickle=False
)
model2.train(
    context_size=3, 
    embedding_size=10, 
    hidden_size=300, 
    epochs=epochs,
    regularization_strength=0.00001,
    learning_rate=0.5,
    batch_size=500,
    pickle=False,
    weight_adjustments=[0.01, 0, 0.01, 0]
)
# print(model1.pre_activations)
pa_tensor = torch.tensor(model1.pre_activations).view(1, -1).tolist()
a_tensor = torch.tensor(model1.activations).view(1, -1)
tail_perc = (a_tensor.abs() > 0.99).sum()/a_tensor.count_nonzero() * 100
a_tensor_list = torch.tensor(model1.activations).view(1, -1).tolist()

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])  # 2 rows, 2 cols, bottom row smaller

# Hist 1 (top-left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(pa_tensor, bins=35, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('Pre-Activations')
ax1.axvline(x=-2.65, color='red', linestyle='-', label='Tanh Tail Boundary')
ax1.axvline(x=2.65, color='red', linestyle='-')
ax1.plot([], [], ' ', label=f'~{tail_perc:.2f}% Tail Region')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.5)

# Hist 2 (top-right)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(a_tensor_list, bins=35, color='lightgreen', edgecolor='black', alpha=0.7)
ax2.set_title('Activations')
ax2.grid(True, linestyle='--', alpha=0.5)

ac_img = (torch.tensor(model1.activations).abs() > 0.99)
zero_cols = (ac_img.sum(dim=0) == 0).nonzero(as_tuple=True)[0]

ax3 = fig.add_subplot(gs[1, :])  # span both columns
ax3.imshow(ac_img.tolist(), cmap='gray', aspect='auto')
ax3.set_xlabel("Hidden Parameter Index")
ax3.set_ylabel("Example Number")

for col in zero_cols.tolist():
    rect = plt.Rectangle(
        (col - 0.5, -0.5), 1, ac_img.shape[0],  # (x, y), width, height
        linewidth=20, edgecolor='red', facecolor='none'
    )
    ax3.add_patch(rect)
# Grid for better readability

# Show plot
plt.tight_layout()
plt.savefig("/app/3_makemore_v2/figures/random_vs_adjusted_weight_pre_activations.png", dpi=300, bbox_inches="tight")

# plot_loss_compare(model1, model2)


# model.eval()
# learning_rates, losses = model.tune_learning_rate(
#     steps=1000,
#     context_size=3, 
#     embedding_size=2, 
#     hidden_size=100, 
#     batch_size=100
# )

# variances = []
# slopes = []

# window_size = 100
# for chunk_start, chunk_end in zip(range(0, 1000, window_size), range(window_size, 1000 + window_size, window_size)):
#     slope, b = np.polyfit(learning_rates[chunk_start:chunk_end], losses[chunk_start:chunk_end], 1)
#     var = np.var(losses[chunk_start:chunk_end])
#     variances.append(var)
#     slopes.append(slope / var)

# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Left subplot: NLL vs Learning Rate
# axes[0].plot(learning_rates, losses, linestyle='-')
# axes[0].set_title('Mean NLL vs. Learning Rate', fontsize=14)
# axes[0].set_xlabel('Learning Rate', fontsize=12)
# axes[0].set_ylabel('Mean Negative Log Likelihood', fontsize=12)
# axes[0].grid(True)

# # Right subplot: Slope vs Variance Window
# x2 = list(range(window_size, 1000 + window_size, window_size))
# axes[1].plot(x2, slopes, linestyle='-', marker='o')
# axes[1].set_title('Slope vs. Variance Window', fontsize=14)
# axes[1].set_xlabel('Window', fontsize=12)
# axes[1].set_ylabel('Slope of NLL Variation', fontsize=12)
# axes[1].grid(True)

# # Overall figure title and layout adjustments
# fig.suptitle('Negative Log Likelihood Analysis', fontsize=16)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# # Save to file
# plt.savefig("/app/3_makemore_v2/figures/nll_analysis_combined.png", dpi=300, bbox_inches="tight")
# plt.show()


# model.predict(n=10)

# model.explore()
