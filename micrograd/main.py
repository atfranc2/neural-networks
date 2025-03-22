import math
import numpy as np
import random
from graphviz import Digraph


class Value:
    """
    In local _backwards functions they use +=. This is to handle the case of operating on a Value object
    that are the same (e.g a = Value(1.0) and then having b = a + a). If the gradient was as just = then
    the operations would override each other. Per chain rule these can and should be added together.
    As proof we can justify this using b = a + a = 2a so db/da = 2. 

    Some operations are defined in terms of other operations. For example subtraction is a combination of
    multiplication and addition. Division is defined in terms of multiplication and powers. This was done
    for convenience because operations for __mul__, __pow__, and __add__ already have defined rules for 
    creation and backpropagation.
    """
    def __init__(
            self, 
            data:float, 
            children:tuple['Value']|None=None, 
            operation:str|None=None, 
            gradient:float=0.0, 
            label:str|None=None
        ):
        self.data = data
        self.children = children
        self.operation = operation
        self.gradient = 0.0
        self.label = label

        self._backwards = lambda: None

    def __repr__(self):
        return f"<Value label={self.label} data={self.data} grad={self.gradient} op={self.operation}>"
    
    @property
    def has_children(self):
        if self.children is None:
            return 

        return len(self.children) > 0
    
    def __to_value(self, value:'Value|int|float'):
        return value if isinstance(value, Value) else Value(value)
    
    def __add__(self, other:'Value|int|float'):
        other = self.__to_value(other) # Allows Value + int or Value + Value
        out = Value(self.data + other.data, (self, other), operation="+")
        def backwards():
            self.gradient += out.gradient
            other.gradient += out.gradient

        out._backwards = backwards
        
        return out
    
    def __radd__(self, other:'Value|int|float'):
         # Allows int + Value as Value + int which is supported in __add__
        return self + other
    
    def __neg__(self):
        return self * -1 # Supported through __mul__
    
    def __sub__(self, other:'Value'):
        return self + (-other) # Define subtraction in terms of __add__ and __mul__ (from neg as self * -1)
    
    def __mul__(self, other:'Value|int|float'):
        other = self.__to_value(other)
        out = Value(self.data * other.data, (self, other), operation="*")

        def backwards():
            self.gradient += other.data * out.gradient
            other.gradient +=self.data * out.gradient

        out._backwards = backwards

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other:'Value'):
        return self * other**-1 # Define division in terms of already supported functions of __mul__ and __pow__
    
    def __pow__(self, power:int|float):
        assert isinstance(power, (int, float))
        out = Value(self.data**power, children=(self,), operation=f"pow({power})")

        def backwards():
            self.gradient += power * self.data**(power-1) * out.gradient

        out._backwards = backwards

        return out
    
    def exp(self):
        out = Value(math.exp(self.data), children=(self,), operation="exp")

        def backwards():
            # d/dx e**x = e**x
            self.gradient += out.data * out.gradient

        out._backwards = backwards

        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data), children=(self,), operation="tanh")

        def backwards():
            self.gradient += (1 - math.tanh(self.data)**2) * out.gradient

        out._backwards = backwards
        
        return out
    
    def adjust(self, learn_rate:float):
        """After backpropagation occurs this is used by trainer to adjust the weights and bias"""
        self.data += (self.gradient * -learn_rate)

    def loss(self, target:'Value'):
        target = self.__to_value(target)
        out = Value(0.5 * (target.data - self.data)**2, (self,), operation="loss")

        def backwards():
            self.gradient += (self.data - target.data) * out.gradient

        out._backwards = backwards

        return out
    
    def __build_topo(self, value:"Value", seen:set, topo:list):
        if value not in seen:
            seen.add(value)

            # Drill down into the leaf nodes
            if value.has_children:
                for child_value in value.children:
                    self.__build_topo(child_value, seen, topo)

            topo.append(value)


    def backwards(self):
        """
        Here we have to visit nodes on a specific order and have to process some nodes before processing
        other nodes. Enter topological sort:
            - Algorithm used to sort and process dependencies
            - Only works on directed data structures without a cycle
            - Can be implemented using depth first traversal and a stack. You do not add an item to the stack
            until you have visited all the children of that node
        """
        self.gradient = 1.0
        seen = set()
        topo = []
        self.__build_topo(self, seen, topo)
        for node in reversed(topo):
            node._backwards()

        """
        o, 0.7071067811865477
        n, 0.8813735870195432
        b1, 6.881373587019543
        x1w1 + x2w2, -6.0
        x2*w2, 0.0
        w2, 1.0
        x2, 0.0
        x1*w1, -6.0
        w1, -3.0
        x1, 2.0
        """


class Neuron:
    def __init__(self, input_size:int):
        self.input_size = input_size
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1, 1))

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]

    def __call__(self, inputs:list[Value]):
        outputs = []
        for weight, input in zip(self.weights, inputs):
            outputs.append(weight * input)

        activation = sum(outputs) + self.bias
        
        return activation.tanh()
    

class Layer:
    def __init__(self, num_inputs:int, num_outputs:int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def parameters(self) -> list[Value]:
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

    def __call__(self, inputs:list[Value]):
        return [neuron(inputs) for neuron in self.neurons]
    

class MultiLayerPreceptron:
    def __init__(self, input_size:int, layer_sizes:list[int]):
        self.layers:list[Layer] = []
        __input_size = input_size
        for output_size in layer_sizes:
            self.layers.append(Layer(__input_size, output_size))
            __input_size = output_size

    def parameters(self) -> list[Value]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.gradient = 0.0

    def adjust(self, learn_rate:float):
        for param in self.parameters():
            param.adjust(learn_rate)
                
    def __call__(self, inputs:list[Value]):
        __inputs = inputs
        for layer in self.layers:
            __inputs = layer(__inputs)

        return __inputs
    

class Train:
    def __init__(self, mlp:MultiLayerPreceptron):
        self.mlp = mlp

    def __call__(self, training:list[list], targets:list, learn_rate:float, epochs:int):
        for _ in range(epochs):
            predictions = [mlp(inputs)[0] for inputs in training]
            loss:Value = sum([(prediction - target)**2 for prediction, target in zip(predictions, targets)])
            # You always have to zero out your parameter gradients before backpropogating so that your
            # Gradients do not accumulate. Instead they will reflect the gradient with respect to the 
            # updated network function across epochs. 
            mlp.zero_grad()
            loss.backwards()
            mlp.adjust(learn_rate)
            # pairs = [f"{(target, prediction)}" for prediction, target in zip(predictions, targets)]
            # print(f"loss:{loss.data}, Pairs{pairs}")

        return loss


def _draw_graph(value:Value, prev_value:Value|None, diagram:Digraph, seen_value:set, seen_edge:set):
    node_label = str(id(value))
    prev_label = str(id(prev_value))

    # We have reached a leaf node
    if not value.has_children:
        if value in seen_value:
            return
        
        operation_label = prev_label + "-operation"
        diagram.node(operation_label, label=f"{prev_value.operation}", shape="circle")
        if (operation_label, prev_label) not in seen_edge:
            diagram.edge(operation_label, prev_label)
            seen_edge.add((operation_label, prev_label))
        
        diagram.node(node_label, label="{ Value: %.4f | Label: %s | Gradient: %.4f }" % (value.data, value.label, value.gradient), shape="record")
        if (node_label, operation_label) not in seen_edge:
            diagram.edge(node_label, operation_label)
            seen_edge.add((node_label, operation_label))

        seen_value.add(value)

        return
    
    # Drill down into the leaf nodes
    for child_value in value.children:
        _draw_graph(child_value, value, diagram, seen_value, seen_edge)
    
    if value in seen_value:
        return
    
    diagram.node(node_label, label="{ Value: %.4f | Label: %s | Gradient: %.4f }" % (value.data, value.label, value.gradient), shape="record")
    
    if prev_value is not None:
        operation_label = prev_label + "-operation"
        diagram.node(operation_label, label=f"{prev_value.operation}", shape="circle")
        if (operation_label, prev_label) not in seen_edge:
            diagram.edge(operation_label, prev_label)
            seen_edge.add((operation_label, prev_label))
        if (node_label, operation_label) not in seen_edge:
            diagram.edge(node_label, operation_label)
            seen_edge.add((node_label, operation_label))

    seen_value.add(value)

def draw_graph(value:Value):
    seen_value = set()
    seen_edge = set()
    g = Digraph(format="png", graph_attr={'rankdir': 'LR'})  # Specify output format
    _draw_graph(value, None, g, seen_value, seen_edge)
    g.render("/app/micrograd/figures/graph")

x = [Value(2.0), Value(3.0), Value(-1.0)]
xs = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]
ys = [1.0, -1.0, -1.0, 1.0]
mlp = MultiLayerPreceptron(3, [4,4,1])
losses = [(mlp(inputs)[0] - target)**2 for inputs, target in zip(xs, ys)]
loss = sum(losses)
print("Pre-Train Avg Loss:", loss)

trainer = Train(mlp)
loss = trainer(xs, ys, 0.99, 100)

print("Post-Train Avg Loss:", loss)
# print(mlp.parameters())
# print(loss)
# draw_graph(loss)

x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

b1 = Value(6.8813735870195432, label="b1")
b2 = Value(7.8813735870195432, label="b2")

# x1w1 = x1*w1
# x1w1.label = "x1*w1"

# x2w2 = x2*w2
# x2w2.label = "x2*w2"

# x1w1_x2w2 = x1w1+x2w2
# x1w1_x2w2.label = "x1w1 + x2w2"

# n=x1w1_x2w2+b1
# n.label = "n"

# e = (2*n).exp()
# o = (e - 1) / (e + 1)
# # o = n.tanh()
# o.label = "o"

# o.backwards()
# draw_graph(o)


# y1 = x1 + w1
# y1.label = "y1"

# y2 = y1 + b1
# y2.label = "y2"

# y2_a = y2 + b2
# y2_a.label = "y2_a"
# x1_w1 = x1*w1
# x1_w1.label = "x1*w1"
# y1 = x1_w1 + b1
# y1.label = "y1"
# y1_a = y1.tanh()
# y1_a.label = "y1_a"

# y1_a_w2 = y1_a*w2
# y1_a_w2.label = "y1_a*w2"
# y2 = y1_a_w2 + b2
# y2.label = "y2"
# y2_a = y2.tanh()
# y2_a.label = "y2_a"

# target1 = Value(2, label="target1")
# loss = y2_a.loss(target1)
# loss.label = "loss"


# draw_graph(o)
# a = Value(3.0, label="a")
# b = a + a
# b.label = "b"

# # b.backwards()
# draw_graph(b)