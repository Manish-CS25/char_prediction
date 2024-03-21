import streamlit as st
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# from pprint import pprint

from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
with open('pg.txt', 'r') as file:
    text = file.read()

words = text.split()




# Filter the list of words based on their length
words = [word for word in words if 2 < len(word) < 10]



# # Randomly shuffle the words
# words = words.sample(frac=1).reset_index(drop=True)
# words = words.tolist()

# Remove words having non alphabets
words = [word for word in words if word.isalpha()]
words[:10]


# Convert the text into a list of characters
chars = list(text)

# Convert to lowercase
chars = [char.lower() for char in chars]

# Filter out non-alphabet characters
chars = [char for char in chars if char.isalpha()]

# Randomly shuffle the characters
np.random.shuffle(chars)

# Build the vocabulary of characters and mappings to/from integers
# Assuming `text` is your data and `stoi` is your string-to-integer mapping
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
print(itos)


block_size = 5 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:]:

  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append

# Move data to GPU

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)


emb_dim = 4
emb = torch.nn.Embedding(len(stoi), emb_dim)





def plot_emb(emb, itos, ax=None):
    # Get the weights of the embedding layer
    weights = emb.weight.detach().cpu().numpy()

    # Use PCA to reduce the dimensionality to 2
    pca = PCA(n_components=2)
    weights_pca = pca.fit_transform(weights)

    if ax is None:
        fig, ax = plt.subplots()

    for i in range(len(itos)):
        x, y = weights_pca[i]
        ax.scatter(x, y, color='k')
        ax.text(x + 0.05, y + 0.05, itos[i])

    return ax

plot_emb(emb, itos)

class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
# Generate names from untrained model


model = NextChar(block_size, len(stoi), emb_dim, 10).to(device)
model = torch.compile(model)


# Use FX for symbolic tracing
traced_model = torch.fx.symbolic_trace(model)

# Optimize with Dynamo
optimized_model = dynamo.optimize(traced_model)

# Use the optimized model
y_pred = optimized_model(x)

g = torch.Generator()
g.manual_seed(4000002)
def generate_name(model, itos, stoi, block_size, max_len=10):
    context = [0] * block_size
    name = ''
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '.':
            break
        name += ch
        context = context[1:] + [ix]
    return name

# for i in range(10):
#     print(generate_name(model, itos, stoi, block_size))

# Train the model

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
import time
# Mini-batch training
batch_size = 4096
print_every = 100
elapsed_time = []
for epoch in range(1000):
    start_time = time.time()
    for i in range(0, X.shape[0], batch_size):
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    end_time = time.time()
    elapsed_time.append(end_time - start_time)
    if epoch % print_every == 0:
        print(epoch, loss.item())

# for i in range(10):
#     print(generate_name(model, itos, stoi, block_size))











# Interface
st.markdown("## Next k character prediction app")

# Sliders for context size and embedding dimension
context_size = st.slider("Number of k characters prediction", min_value=1, max_value=10, value=5)
emb_dim = st.slider("Embedding dimension", min_value=1, max_value=100, value=50)

# Text input for next character prediction
text_input = st.text_input("Enter text for next character prediction")

# Predict button
if st.button("Predict"):
    # Create a new model with the user-specified embedding 


    
    # Define your model
    model = NextChar(context_size, len(stoi), emb_dim, 10).to(device)

# Convert the model to TorchScript
    scripted_model = torch.jit.script(model)

# Use the scripted model for prediction
    prediction = generate_name(scripted_model, itos, stoi, context_size)
    st.write(prediction)
