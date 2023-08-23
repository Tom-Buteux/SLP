import pickle as pkl
import pandas as pd
import numpy as np

# load cat_data
cat_data = pd.read_pickle('cat_data.pkl')

# load quads and hashcodes from npy files
quads = np.load('quads.npy', allow_pickle=True)
hashcodes = np.load('hashcodes.npy', allow_pickle=True)

print('Number of quads: ', len(quads))
print('Number of hashcodes: ', len(hashcodes))

print(cat_data)