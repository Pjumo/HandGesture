import numpy as np

for i in range(10):
    data = np.load(f'dataset/doubleclick/seq_{i+1}.npy')
    np.set_printoptions(threshold=np.inf)
    print(data.shape)
