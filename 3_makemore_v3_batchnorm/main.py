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


class Layer:
    def __init__(self, on_apply):
        self.on_apply = on_apply

    def parameters(self) -> list[torch.Tensor]:
        raise NotImplementedError()
    
    def _apply(self, tensor:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def apply(self, tensor:torch.Tensor) -> torch.Tensor:
        self.output = self._apply(tensor)
        if self.on_apply:
            self.on_apply(self, self.output)

        return self.output

    def require_grad(self):
        for parameter in self.parameters():
            parameter.requires_grad = True

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None


class Softmax(Layer):
    def parameters(self):
        return []
    
    def _apply(self, tensor:torch.Tensor):
        self.output = torch.softmax(tensor)
        return self.output


class NegativeLogLikelihood(Layer):
    def __init__(self, target:torch.Tensor):
        super().__init__()
        self.target = target
    
    def parameters(self):
        return []
    
    def _apply(self, tensor):
        self.output = F.cross_entropy(tensor, self.target)
        return self.output


class Linear(Layer):
    def __init__(self, in_features, out_features, gain=1, on_apply=None, do_kaiming=True, bias=True):
        super().__init__(on_apply)
        self.gain = gain
        kaiming = (gain/in_features)**0.5 if do_kaiming else 1
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.randn(in_features, out_features) * torch.tensor(kaiming)
        self.b = torch.zeros(1, out_features) * kaiming if bias else None

    def parameters(self):
        params = [self.W]
        if self.b is not None:
            params.append(self.b)

        return params

    def _apply(self, other:torch.Tensor):
        self.output = other @ self.W
        if self.b is not None:
            self.output += self.b

        return self.output


class Tanh(Layer):
    def __init__(self, on_apply=None):
        super().__init__(on_apply)

    def parameters(self):
        return []
    
    def _apply(self, input:torch.Tensor):
        self.output = torch.tanh(input)
        return self.output


class BatchNorm1D(Layer):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, on_apply=None, track_running_stats=True):
        super().__init__(on_apply)
        self.training = True
        self.num_features = num_features
        self.eps = eps

        # Affine Parameters
        self.affine = affine
        self.shift = torch.zeros(1, num_features) # Beta
        self.scale = torch.ones(1, num_features) # Gamma
        
        # Running Stats (buffers)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.running_mean = torch.zeros(1, num_features)
        self.running_var = torch.ones(1, num_features)

    def parameters(self):
        return [self.shift, self.scale]

    def _apply(self, tensor:torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = tensor.mean(0, keepdim=True)
            var = tensor.var(0, keepdim=True)
        else:
            mean = self.running_mean
            var = self.running_var

        if self.track_running_stats and self.training:
            with torch.no_grad():
                self.running_mean = self.running_mean * (1 - self.momentum) + mean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + var * self.momentum

        normalized = (tensor - mean) / torch.sqrt(var + self.eps)
        if self.affine: 
            normalized = self.scale * normalized + self.shift
        
        self.out = normalized
        
        return normalized


class Network:
    def __init__(self, train_in:torch.Tensor, train_out:torch.Tensor, layers:list[Layer]):
        self.train_in = train_in
        self.train_out = train_out
        self.input = layers[0]
        self.layers = layers
        self.initialized = False

    def parameters(self) -> list[torch.Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]
    
    def require_grad(self):
        for layer in self.layers:
            layer.require_grad()

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def adjust_grad(self, learning_rate):
        for parameter in self.parameters():
            parameter.data += parameter.grad * -learning_rate

    def forwards(self, batch_size:int):
        if not self.initialized:
            self.require_grad()
            self.initialized = True
        
        self.batch_indexes = torch.randint(0, self.train_in.shape[0], (batch_size,))
        batch_examples = self.train_in[self.batch_indexes]
        batch_input = self.input.W[batch_examples]
        self.output = batch_input.view(batch_size, -1)
        for layer in self.layers[1:]:
            self.output = layer.apply(self.output)

    def backwards(self, learning_rate=0.1):
        loss = F.cross_entropy(self.output, self.train_out[self.batch_indexes])
        self.zero_grad()
        loss.backward()

        return loss
        

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
        batch_size = 500
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
            # print("training_examples:", training_examples.shape)


            # A three dimensional matrix. First dim: examples, Second dim: word embedding vectors, Third Dim: Embeddings 
            # Size: batch_size, context_size, embedding_size
            batch = self.C[training_examples]
            # print("batch:", batch.shape)
            # print("batch*:", batch.view(batch_size, -1).shape)

            pre_activation = batch.view(batch_size, -1) @ self.W1
            # print("pre_activation:", pre_activation.shape)
            # Sum down the columns (vertically)
            # Takes mean + std of all pre-activations across batch
            batch_mean = pre_activation.mean(0, keepdim=True) # Sum down the columns (vertically)
            batch_std = pre_activation.std(0, keepdim=True)
            # print("batch_mean:", batch_mean.shape)

            with torch.no_grad():
                self.bn_running_mean = 0.9 * self.bn_running_mean + 0.1 * batch_mean
                self.bn_running_std = 0.9 * self.bn_running_std + 0.1 * batch_std

            bn_pre_activation = self.bn_scale * ((pre_activation - batch_mean) / batch_std) + self.bn_shift
            activation = torch.tanh(bn_pre_activation)
            # print("activation:", activation.shape)

            logits = activation @ self.W2 + self.b2
            loss = F.cross_entropy(logits, self.Y[batch_indexes])
            # print("logits:", logits.shape)
            # print("loss:", loss.shape)

            for parameter in parameters: 
                parameter.grad = None

            # Capture snapshot every 500 iterations (node-75 distributions + BN params)
            if epoch % 500 == 0:
                with torch.no_grad():
                    node_idx = 5  # 0-based index for the 75th node
                    raw_node = pre_activation[:, node_idx].detach().cpu().flatten().numpy()
                    norm_full = self.bn_scale * ((pre_activation - batch_mean) / batch_std) + self.bn_shift
                    norm_node = norm_full[:, node_idx].detach().cpu().flatten().numpy()
                    raw_mu = float(np.mean(raw_node)); raw_sd = float(np.std(raw_node))
                    norm_mu = float(np.mean(norm_node)); norm_sd = float(np.std(norm_node))
                    self._snapshots.append({
                        'epoch': epoch,
                        'node_idx': node_idx,
                        'raw_node': raw_node,
                        'norm_node': norm_node,
                        'raw_mu': raw_mu,
                        'raw_sd': raw_sd,
                        'norm_mu': norm_mu,
                        'norm_sd': norm_sd,
                        'node_scale': float(self.bn_scale[0, node_idx].item()),
                        'node_shift': float(self.bn_shift[0, node_idx].item()),
                    })

            loss.backward()
            
            for parameter in parameters:
                # print(parameter.grad)
                parameter.data += parameter.grad * -learning_rate

        self.loss = loss.data

        # Plotting/animation is handled outside the class after training
        
    def train_v2(
        self, 
        layers,
        scenario,
        do_kaiming,
        plot_epoch=0,
        epochs:int = 10000,
    ):
        # MLP Reference
        # BatchNorm Reference - https://arxiv.org/abs/1502.03167
        
        learning_rate = 1.5
        batch_size = 32
        hidden_size = 100
        embedding_size = 10
        self.context_size = 3

        self.train_words, self.validate_words, self.test_words = self.split(self.words)
        
        # Each x is a vector of length context size. Each element containing the character index
        self.X, self.Y = self.n_gram_tensors(self.train_words, self.context_size)

        self.epoch = 0
        linear_activation_figure = None
        linear_activation_ax = None

        norm_activation_figure = None
        norm_activation_ax = None

        tanh_activation_figure = None
        tanh_activation_ax = None

        linear_gradient_figure = None
        linear_gradient_ax = None

        tanh_gradient_figure = None
        tanh_gradient_ax = None

        layer_colors = {}
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
        def update_histogram(layer:Layer, output:torch.Tensor, ax, title, legend=None):
            if plot_epoch != self.epoch:
                return
            
            _legend = [] if not legend else legend
            # Assign a stable color per layer instance
            lid = id(layer)
            if lid not in layer_colors:
                layer_colors[lid] = color_cycle[len(layer_colors) % len(color_cycle)]
            color = layer_colors[lid]

            # Compute stats and plot depending on layer type
            if isinstance(layer, Tanh):
                data = output.detach().cpu().view(-1)
                mu = float(data.mean().item())
                sd = float(data.std(unbiased=False).item())
                sat = int((data.abs() > 0.97).sum().item()) / len(data)
                counts, bin_edges = np.histogram(data.numpy(), bins=100)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(bin_centers, counts, color=color,
                        label=f"Tanh: μ={mu:.4f}, σ={sd:.4f}, sat={sat:.4f}")
            elif isinstance(layer, (Linear, BatchNorm1D)):
                # For Linear, characterize parameters W statistics
                data = output.detach().cpu().view(-1)
                mu = float(data.mean().item())
                sd = float(data.std(unbiased=False).item())
                counts, bin_edges = np.histogram(data.numpy(), bins=100)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(bin_centers, counts, color=color,
                        label=f"Linear: μ={mu:.4f}, σ={sd:.4f}, {','.join(_legend)}")
            elif isinstance(layer, torch.Tensor):
                # For Linear, characterize parameters W statistics
                data = output.detach().cpu().view(-1)
                mu = float(data.mean().item())
                sd = float(data.std(unbiased=False).item())
                counts, bin_edges = np.histogram(data.numpy(), bins=100)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(bin_centers, counts, color=color,
                        label=f"Linear: μ={mu:.4f}, σ={sd:.4f}, {','.join(_legend)}")
            else:
                return

            ax.set_xlabel("value")
            ax.set_ylabel("frequency")
            ax.set_title(title)
            ax.legend(frameon=False)
        
        _layers = []
        C = Linear(self.n_chars, embedding_size, do_kaiming=False, bias=False)
        for layer in layers:
            if isinstance(layer, Linear):
                if linear_activation_figure is None:
                    linear_activation_figure = plt.figure(1, (10, 6))
                    linear_activation_ax = linear_activation_figure.gca()
                layer.on_apply = lambda layer, output: update_histogram(
                    layer, 
                    output, 
                    linear_activation_ax,
                    f"Linear Pre-Activation"
                )
            elif isinstance(layer, Tanh):
                if tanh_activation_figure is None:
                    tanh_activation_figure = plt.figure(2, (10, 6))
                    tanh_activation_ax = tanh_activation_figure.gca()
                layer.on_apply = lambda layer, output: update_histogram(
                    layer, 
                    output, 
                    tanh_activation_ax,
                    f"Tanh Activation"
                )
            elif isinstance(layer, BatchNorm1D):
                if norm_activation_figure is None:
                    norm_activation_figure = plt.figure(3, (10, 6))
                    norm_activation_ax = norm_activation_figure.gca()
                layer.on_apply = lambda layer, output: update_histogram(
                    layer, 
                    output, 
                    norm_activation_ax,
                    f"BatchNorm Pre-Activation"
                )
            _layers.append(layer)

        self.network = Network(self.X, self.Y, [
            C,
            *_layers,
            Linear(hidden_size, self.n_chars, gain=5/3, do_kaiming=do_kaiming), Tanh(),
        ])

        weight_adjustments = [[] for layer in self.network.parameters()]
        for epoch in range(epochs):
            self.epoch = epoch
            if epoch % 100 == 0: 
                print(f"Epoch {epoch}/{epochs}")
            self.network.forwards(batch_size)
            for layer in _layers:
                layer.output.retain_grad()
            self.loss = self.network.backwards()
            self.network.adjust_grad(learning_rate)

            with torch.no_grad():
                for i, parameter in enumerate(self.network.parameters()):
                    if 1 in parameter.shape:
                        continue

                    if linear_gradient_figure is None:
                        linear_gradient_figure = plt.figure(4, (10, 6))
                        linear_gradient_ax = linear_gradient_figure.gca()

                    update_histogram(
                        parameter, 
                        parameter.grad, 
                        linear_gradient_ax, 
                        "Weight Gradient Distribution",
                        [f'grad:data = {(parameter.grad.detach().cpu() / parameter.data.detach().cpu()).mean().item():.4f}']
                    ) 

                for layer in _layers:
                    if epoch != plot_epoch:
                        continue

                    if not isinstance(layer, Tanh):
                        continue

                    if tanh_gradient_figure is None:
                        tanh_gradient_figure = plt.figure(55, (10, 6))
                        tanh_gradient_ax = tanh_gradient_figure.gca()

                    update_histogram(
                        layer, 
                        layer.output.grad, 
                        tanh_gradient_ax, 
                        "Tanh Gradient Distribution",
                        # [f'grad:data = {(parameter.grad.detach().cpu() / parameter.data.detach().cpu()).mean().item():.4f}']
                    )             

            for i, parameter in enumerate(self.network.parameters()):
                weight_adjustments[i].append(((learning_rate*parameter.grad).std() / parameter.data.std()).log10().item())

        if norm_activation_figure:
            norm_activation_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_norm_activation_hist.png")
            plt.close(norm_activation_figure)
        
        if linear_activation_figure:
            linear_activation_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_linear_activation_hist.png")
            plt.close(linear_activation_figure)
        
        if tanh_activation_figure:
            tanh_activation_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_tanh_activation_hist.png")
            plt.close(tanh_activation_figure)
        
        if linear_gradient_figure:
            linear_gradient_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_linear_gradient_hist.png")
            plt.close(linear_gradient_figure)
        
        if tanh_gradient_figure:
            tanh_gradient_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_tanh_gradient_hist.png")
            plt.close(tanh_gradient_figure)

        weight_adjustment_figure = plt.figure(6, figsize=(20, 4))
        weight_adjustment_ax = weight_adjustment_figure.gca()
        
        legends = []
        for i, parameter in enumerate(self.network.parameters()):
            if len(parameter) == 0:
                continue
            
            if 1 in parameter.shape:
                continue
            
            weight_adjustment_ax.plot(weight_adjustments[i])
            legends.append('param %d' % i)

        weight_adjustment_ax.plot([0, len(weight_adjustments[0])], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        weight_adjustment_ax.legend(legends)

        weight_adjustment_ax.set_ylim(-5, 0)
        weight_adjustment_figure.savefig(f"3_makemore_v3_batchnorm/figures/{scenario}_learning_progression.png")
        plt.close(weight_adjustment_figure)
        
        return self.loss
    
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
    

# python /app/3_makemore_v3_batchnorm/main.py
hidden_size = 100
embedding_size = 10
context_size = 3
epochs = 1000
plot_epoch = 999
gain = 5/3


raw_model = NeuralModel()
raw_layers = [
    Linear(context_size * embedding_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=True), Tanh(),
]
raw_model.train_v2(raw_layers, 'raw', do_kaiming=False, epochs=epochs, plot_epoch=plot_epoch)


kaiming_model = NeuralModel()
kaiming_layers = [
    Linear(context_size * embedding_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=True), Tanh(),
]
kaiming_model.train_v2(kaiming_layers, 'kaiming', do_kaiming=True, epochs=epochs, plot_epoch=plot_epoch)


batchnorm_model = NeuralModel()
batchnorm_layers = [
    Linear(context_size * embedding_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=False, bias=False), BatchNorm1D(hidden_size), Tanh(),
]
batchnorm_model.train_v2(batchnorm_layers, 'batchnorm', do_kaiming=False, epochs=epochs, plot_epoch=plot_epoch)


batchnorm_kaiming_model = NeuralModel()
batchnorm_kaiming_layers = [
    Linear(context_size * embedding_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
    Linear(hidden_size, hidden_size, gain=gain, do_kaiming=True, bias=False), BatchNorm1D(hidden_size), Tanh(),
]
batchnorm_kaiming_model.train_v2(batchnorm_kaiming_layers, 'batchnorm_kaiming', do_kaiming=True, epochs=epochs, plot_epoch=plot_epoch)

# Build animation outside the class: node-75 raw vs normalized histograms
# if len(model._snapshots) > 0:
if False:
    # Global limits per distribution type (raw vs normalized)
    all_raw = np.concatenate([s['raw_node'] for s in model._snapshots])
    all_norm = np.concatenate([s['norm_node'] for s in model._snapshots])
    rmin, rmax = float(all_raw.min()), float(all_raw.max())
    nmin, nmax = float(all_norm.min()), float(all_norm.max())
    rpad = 0.05 * (rmax - rmin + 1e-8)
    npad = 0.05 * (nmax - nmin + 1e-8)
    rmin -= rpad; rmax += rpad
    nmin -= npad; nmax += npad
    r_bins = np.linspace(rmin, rmax, 75)
    n_bins = np.linspace(nmin, nmax, 75)

    # 2x2 layout: top row histograms, bottom row lines of mean/std across epochs
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1]})
    ax_raw, ax_norm = axes[0]
    ax_mu, ax_sd = axes[1]

    # Precompute full time-series once for static bottom row
    epochs = np.array([s['epoch'] for s in model._snapshots])
    raw_mus = np.array([s.get('raw_mu', float(np.mean(s['raw_node']))) for s in model._snapshots])
    raw_sds = np.array([s.get('raw_sd', float(np.std(s['raw_node']))) for s in model._snapshots])
    norm_mus = np.array([s.get('norm_mu', float(np.mean(s['norm_node']))) for s in model._snapshots])
    norm_sds = np.array([s.get('norm_sd', float(np.std(s['norm_node']))) for s in model._snapshots])

    def update_hist(i):
        snap = model._snapshots[i]
        ax_raw.cla(); ax_norm.cla()

        mu_r = float(np.mean(snap['raw_node']))
        sd_r = float(np.std(snap['raw_node']))
        ax_raw.hist(snap['raw_node'], bins=r_bins, color='tab:blue', alpha=0.75,
                    label=f"Raw: μ={mu_r:.4f}, σ={sd_r:.4f}")
        ax_raw.set_title(f"Node {snap['node_idx']+1} raw — epoch {snap['epoch']}")
        ax_raw.set_xlabel("value"); ax_raw.set_ylabel("count")
        ax_raw.set_xlim(rmin, rmax)
        ax_raw.legend(frameon=False)

        mu_n = float(np.mean(snap['norm_node']))
        sd_n = float(np.std(snap['norm_node']))
        gamma = snap['node_scale']; beta = snap['node_shift']
        ax_norm.hist(snap['norm_node'], bins=n_bins, color='tab:orange', alpha=0.75,
                     label=f"Norm: μ={mu_n:.4f}, σ={sd_n:.4f}\nscale={gamma:.4f}, shift={beta:.4f}")
        ax_norm.set_title(f"Node {snap['node_idx']+1} normalized — epoch {snap['epoch']}")
        ax_norm.set_xlabel("value"); ax_norm.set_ylabel("count")
        ax_norm.set_xlim(nmin, nmax)
        ax_norm.legend(frameon=False)

        # Time-series lines always show full data (not animated)
        ax_mu.cla()
        ax_mu.plot(epochs, raw_mus, color='tab:blue', label='Raw μ')
        ax_mu.plot(epochs, norm_mus, color='tab:orange', label='Norm μ')
        ax_mu.set_title(f'Mean over epochs (node {snap['node_idx']+1})')
        ax_mu.set_xlabel('epoch'); ax_mu.set_ylabel('mean')
        ax_mu.grid(alpha=0.2, linestyle=':'); ax_mu.legend(frameon=False)

        ax_sd.cla()
        ax_sd.plot(epochs, raw_sds, color='tab:blue', label='Raw σ')
        ax_sd.plot(epochs, norm_sds, color='tab:orange', label='Norm σ')
        ax_sd.set_title(f'Std over epochs (node {snap['node_idx']+1})')
        ax_sd.set_xlabel('epoch'); ax_sd.set_ylabel('std')
        ax_sd.grid(alpha=0.2, linestyle=':'); ax_sd.legend(frameon=False)

        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, update_hist, frames=len(model._snapshots), interval=1200, blit=False, repeat_delay=1500)
    try:
        snap = model._snapshots[0]['node_idx']+1
        from matplotlib.animation import PillowWriter
        import os
        os.makedirs('3_makemore_v3_batchnorm/figures', exist_ok=True)
        anim.save(f'3_makemore_v3_batchnorm/figures/node_{snap}_activations.gif', writer=PillowWriter(fps=1))
        print("Saved animation to 3_makemore_v3_batchnorm/figures/node75_activations.gif")

        # Save a separate static PNG with the full time-series lines
        fig_lines, (ax_mu_png, ax_sd_png) = plt.subplots(1, 2, figsize=(12, 4))
        ax_mu_png.plot(epochs, raw_mus, color='tab:blue', label='Raw μ')
        ax_mu_png.plot(epochs, norm_mus, color='tab:orange', label='Norm μ')
        ax_mu_png.set_title(f'Mean over epochs (node {snap})')
        ax_mu_png.set_xlabel('epoch'); ax_mu_png.set_ylabel('mean')
        ax_mu_png.grid(alpha=0.2, linestyle=':'); ax_mu_png.legend(frameon=False)

        ax_sd_png.plot(epochs, raw_sds, color='tab:blue', label='Raw σ')
        ax_sd_png.plot(epochs, norm_sds, color='tab:orange', label='Norm σ')
        ax_sd_png.set_title(f'Std over epochs (node {snap})')
        ax_sd_png.set_xlabel('epoch'); ax_sd_png.set_ylabel('std')
        ax_sd_png.grid(alpha=0.2, linestyle=':'); ax_sd_png.legend(frameon=False)

        fig_lines.tight_layout()
        fig_lines.savefig(f'3_makemore_v3_batchnorm/figures/node_{snap}_activations_stats.png', dpi=150)
        plt.close(fig_lines)
        print(f"Saved static stats figure to 3_makemore_v3_batchnorm/figures/node_{snap}_activations_stats.png")

        # Also save static PNG for node-specific BN shift (beta) and scale (gamma)
        gammas = np.array([s['node_scale'] for s in model._snapshots])
        betas = np.array([s['node_shift'] for s in model._snapshots])
        fig_bn, (ax_g, ax_b) = plt.subplots(1, 2, figsize=(12, 4))
        ax_g.plot(epochs, gammas, color='tab:green')
        ax_g.set_title(f'BN scale γ over epochs (node {snap})')
        ax_g.set_xlabel('epoch'); ax_g.set_ylabel('gamma')
        ax_g.grid(alpha=0.2, linestyle=':')

        ax_b.plot(epochs, betas, color='tab:red')
        ax_b.set_title(f'BN shift β over epochs (node {snap})')
        ax_b.set_xlabel('epoch'); ax_b.set_ylabel('beta')
        ax_b.grid(alpha=0.2, linestyle=':')

        fig_bn.tight_layout()
        fig_bn.savefig(f'3_makemore_v3_batchnorm/figures/node_{snap}_bn_params.png', dpi=150)
        plt.close(fig_bn)
        print(f"Saved BN params figure to 3_makemore_v3_batchnorm/figures/node_{snap}_bn_params.png")
    except Exception as e:
        print("Could not save GIF animation:", e)

# print(f"Train Loss: {model.loss}")
# model.eval("test")
