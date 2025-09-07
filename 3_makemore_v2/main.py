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
        self.C_STEPS = []
        for epoch in range(epochs):
            if epoch % 100 == 0: 
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
                self.pre_activations.append(o1.detach().cpu())
                self.activations.append(o1_activated.detach().cpu())
            
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

            if epoch % 1000 == 0: 
                self.C_STEPS.append(self.C.detach().cpu().numpy().copy())

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
        
    def safe_pop(self, array:list, index=-1, default=None):
        try:
            return array.pop(index)
        except IndexError:
            return default

model = NeuralModel()
model.train(
    context_size=3, 
    embedding_size=10, 
    hidden_size=300,
    epochs=30000,
    regularization_strength=0.00001,
    learning_rate=0.5,
    batch_size=100,
    pickle=True
)

print(f"Train Loss: {model.loss}")
model.eval("test")
model.predict(150)