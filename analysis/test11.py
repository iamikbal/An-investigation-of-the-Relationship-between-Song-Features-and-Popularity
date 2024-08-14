import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data into a numpy array
data = np.genfromtxt('cluster1.csv', delimiter=',', skip_header=1)

# Split the data into features and labels
X = data[:, 3:]
y = data[:, 2]

# Standardize the data
X = StandardScaler().fit_transform(X)

# Create a PCA object with the desired number of components
pca = PCA(n_components=5)

# Fit the PCA model to the data and transform the data into the new space
pca.fit(X)
pca_data = pca.transform(X)

# Adjust the loadings of the first principal component
pca.components_[0][0] = 0.3
pca.components_[0][2] = 0.7

# Transform the data into the new space again
pca_data_adjusted = pca.transform(X)

# Print the first principal component to check that the loadings have been adjusted
print(pca.components_[0])

