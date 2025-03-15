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
    
    def __add__(self, other:'Value'):
        out = Value(self.data + other.data, (self, other), operation="+")
        def backwards():
            self.gradient += out.gradient
            other.gradient += out.gradient

        out._backwards = backwards
        
        return out
    
    def __mul__(self, other:'Value'):
        out = Value(self.data * other.data, (self, other), operation="*")

        def backwards():
            self.gradient += other.data * out.gradient
            other.gradient +=self.data * out.gradient

        out._backwards = backwards

        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data), children=(self,), operation="tanh")

        def backwards():
            self.gradient += (1 - math.tanh(self.data)**2) * out.gradient

        out._backwards = backwards
        
        return out

    def loss(self, target:'Value'):
        out = Value(0.5 * (target.data - self.data)**2, (self,), operation="loss")

        def backwards():
            self.gradient += (self.data - target.data) * out.gradient

        out._backwards = backwards

        return out
    
    def __backwards(self, value:"Value"):
        value._backwards()

        # We have reached a leaf node. 
        if not value.has_children:
            return
        
        # Drill down into the leaf nodes
        for child_value in value.children:
            self.__backwards(child_value)


    def backwards(self):
        self.gradient = 1.0
        self.__backwards(self)


def _draw_graph(value:Value, prev_value:Value|None, diagram:Digraph):
    node_label = str(id(value))
    prev_label = str(id(prev_value))

    # We have reached a leaf node
    if not value.has_children:
        operation_label = prev_label + "-operation"
        diagram.node(operation_label, label=f"{prev_value.operation}", shape="circle")
        diagram.edge(operation_label, prev_label)
        
        diagram.node(node_label, label="{ Value: %.4f | Label: %s | Gradient: %.4f }" % (value.data, value.label, value.gradient), shape="record")
        diagram.edge(node_label, operation_label)

        return
    
    # Drill down into the leaf nodes
    for child_value in value.children:
        _draw_graph(child_value, value, diagram)

    diagram.node(node_label, label="{ Value: %.4f | Label: %s | Gradient: %.4f }" % (value.data, value.label, value.gradient), shape="record")
    
    if prev_value is not None:
        operation_label = prev_label + "-operation"
        diagram.node(operation_label, label=f"{prev_value.operation}", shape="circle")
        diagram.edge(operation_label, prev_label)
        diagram.edge(node_label, operation_label)

def draw_graph(value:Value):
    g = Digraph(format="png", graph_attr={'rankdir': 'LR'})  # Specify output format
    _draw_graph(value, None, g)
    g.render("/app/micrograd/figures/graph")




x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")

b1 = Value(6.8813735870195432, label="b1")
b2 = Value(7.8813735870195432, label="b2")

x1w1 = x1*w1
x1w1.label = "x1*w1"

x2w2 = x2*w2
x2w2.label = "x2*w2"

x1w1_x2w2 = x1w1+x2w2
x1w1_x2w2.label = "x1w1 + x2w2"

n=x1w1_x2w2+b1
n.label = "n"

o = n.tanh()
o.label = "o"

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
a = Value(3.0, label="a")
b = a + a
b.label = "b"

b.backwards()
draw_graph(b)