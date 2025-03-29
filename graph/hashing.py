import hashlib


def hash_to_int(value):
    # Compute SHA-1 hash
    sha1_hash = hashlib.sha1(value.encode()).hexdigest()
    
    # Convert hash to an integer
    return int(sha1_hash, 16)


class Node:
    def __init__(self, identity:str):
        self.identity = identity
        self.position = hash_to_int(identity)

    def __repr__(self):
        return f"<Node id={self.identity}, position={self.position} />"


class Key:
    def __init__(self, identity:str):
        self.identity = identity
        self.membership = None
        self.position = hash_to_int(identity)

    def __repr__(self):
        return f"<identity id={self.identity}, position={self.position} member_of={self.membership} />"


class HashRing:
    def __init__(self):
        self.hash_ring = []
        self.size = 0
        self.node_count = 0
        self.key_count = 0

    def add_node(self, node:Node):
        if self.size == 0:
            self.hash_ring.append(node)

        index = node.position % self.size
        direction = 1 if self.hash_ring[index].position < node.position else -1

    def remove_node(self, node:Node):
        pass

    def add_key(self, key:Key):
        pass

server1 = Node("Server 1")
server2 = Node("Server 2")
server3 = Node("Server 3")
