import numpy as np
from sklearn import random_projection
from sklearn.random_projection import johnson_lindenstrauss_min_dim

print(johnson_lindenstrauss_min_dim(n_samples=30000, eps=0.99))
X = np.random.rand(30000, 128)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)
