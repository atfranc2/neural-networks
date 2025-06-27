# Charachter level language models (makemore)

Note and code based off [lecture series](https://youtu.be/PaCmpygFfXo?si=h85b_svU39TJAeAM)  by Andrej Karpathy

The idea behind makemore is to build out a character level language model that is able to train on a dataset of names and then create new and unique names that it has not seen before. 

The model will work by predicting the next character in the sequence of characters before it. Basically, if you have a name Isabella it may be saying that "s" is likely to come after "i" and that have "a" may indicate the end of the word. 

Basically this will be a weak language model that only has a context window of one character (e.g. the previous character).

```py
import torch

# Initialize a tensor with 3(rows) x 5(columns) storing 32-bit integers
tensor = torch.zeros(3, 5, dtype=torch.int32)

# Can index these as you would a classing matrix 
a[1,3] = 1 # Will set the entry in row 2 column 4 to 1
a[1,3] += 1

# Indexing a tensor returns a tensor
type(a[1,3]) # = Tensor

# To get the integer we call .item()
type(a[1,3].item()) # = int
```
