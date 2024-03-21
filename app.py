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










# def plot_emb(emb, itos, ax=None):
#     # Get the weights of the embedding layer
#     weights = emb.weight.detach().cpu().numpy()
#
#     # Use PCA to reduce the dimensionality to 2
#     pca = PCA(n_components=2)
#     weights_pca = pca.fit_transform(weights)
#
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     for i in range(len(itos)):
#         x, y = weights_pca[i]
#         ax.scatter(x, y, color='k')
#         ax.text(x + 0.05, y + 0.05, itos[i])
#
#     return ax
# plot_emb(emb, itos)
stoi={}
itos={}
words=[]
with open('./processed_data.txt', 'r') as file:
        content = file.read()
        words=content.split(",")
        chars = sorted(list(set(''.join(content.split(",")))))
        stoi = {s:i+1 for i,s in enumerate(chars)}
        stoi['.'] = 0
        itos = {i:s for s,i in stoi.items()}

def generate_name(model,input,k,random_state,block_size):
    g = torch.Generator()
    g.manual_seed(random_state)
    context = [0] * block_size

    name = input
    if len(input)>block_size:
        context=[]
        for i in range(len(input)-block_size,len(input)):
            context.append(stoi[input[i]])
    else:
        for i in range(len(input)):
            context[i+block_size-len(input)]=stoi[input[i]]
    for i in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        name += ch
        context = context[1:] + [ix]
    return name

# for i in range(10):
#     print(generate_name(model, itos, stoi, block_size))




# for i in range(10):
#     print(generate_name(model, itos, stoi, block_size))










# Interface
st.markdown("## Next k character prediction app")

# Sliders for context size and embedding dimension

d1 = st.sidebar.selectbox("Embedding Size", ["2", "5", "10"])
d2 = st.sidebar.selectbox("Context length", ["3", "6", "9"])
d3 = st.sidebar.selectbox("Random state",["4000002","4000005","4000008"])
# Textboxes

t1 = st.sidebar.text_input("Input text", "")
t2 = st.sidebar.text_input("Number of Chars to predict", "")

emb={"2":0,"5":1,"10":2}
context={"3":0,"6":1,"9":2}
# Predict button
if st.button("Predict"):
    # Create a new model with the user-specified embedding
    model1 = NextChar(int(d2),len(stoi), int(d1), 10).to(device)
    model_number=emb[str(d1)]*3+context[str(d2)]
    model1.load_state_dict(torch.load(f"./model_{model_number}.pt"), strict=False)
    model1.eval()
# Use the scripted model for prediction
    prediction = generate_name(model1,t1,int(t2),int(d3),int(d2))
    st.write(prediction)