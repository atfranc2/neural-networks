import string

stuff = ["<S>"] + list(string.ascii_lowercase) + ["<E>"]

for index, char in enumerate(stuff):
    print(index, char)