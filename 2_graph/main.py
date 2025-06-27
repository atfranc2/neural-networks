"""
Providing and example of a topologically sorted graph because we use this principal
when backpropagating a network. 
"""

class Node:
    def __init__(self, data:dict|None=None):
        self.__data = {} if data is None else data


    def __repr__(self):
        return f"<Node data={str(self.__data)} />"

class Graph:
    def __init__(self):
        self.__graph:dict[Node,list[Node]] = {}

    def add_node(self, node:"Node"):
        self.__graph[node] = []

    def add_edge(self, from_node:"Node", to_node:"Node"):
        self.__graph[from_node].append(to_node)

    def get_children(self, node:Node):
        return self.__graph[node]
    

graph = Graph()
a = Node({"label": "A"})
graph.add_node(a)
b = Node({"label": "B"})
graph.add_node(b)
c = Node({"label": "C"})
graph.add_node(c)
d = Node({"label": "D"})
graph.add_node(d)
e = Node({"label": "E"})
graph.add_node(e)
f = Node({"label": "F"})
graph.add_node(f)

graph.add_edge(a, b)
graph.add_edge(a, c)
graph.add_edge(c, d)
graph.add_edge(b, e)
graph.add_edge(b, f)
graph.add_edge(e, d)
graph.add_edge(f, d)

"""
Visual representation of the graph
        F
      /    \ 
  / B - E   
A         \  \
            D
  \      /
    C 
"""

def __topo(graph:Graph, node:Node, visited:set, topo_stack:list):
    # Only visit a node and process its children one time
    if node in visited:
        return
    
    visited.add(node)

    for child in graph.get_children(node):
        __topo(graph, child, visited, topo_stack)

    topo_stack.append(node)


def topo(graph:Graph, root:Node):
    """
    Algorithm, we want to process a node only after we have processed all its children
    and we only want to process the node once. To do this we have to execute a depth first
    traversal. This will create a dependency stack, where we pop the stack to process the nodes
    in correct order
    """
    visited = set()
    topo_stack = []
    __topo(graph, root, visited, topo_stack)

    # "pop" the stack
    # for index, node in enumerate(reversed(topo_stack)):
        # print("Process", node, f"#{index + 1}")


topo(graph, a)