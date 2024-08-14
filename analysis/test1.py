import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Read in the data file containing your audio features for each song in each city and week
data = pd.read_csv('clean-std-data.csv')

# Extract the audio features from the dataset
X = data.iloc[:, 1:]
# print(data)

# Perform PCA on the audio features
pca = PCA(n_components=None)
X_pca = pca.fit_transform(X)

# Print the explained variance ratio for each principal component
print("Explained variance ratio:\n", pca.explained_variance_ratio_)

# Create a Pandas DataFrame for the principal components and their weights
df_pca_components = pd.DataFrame(pca.components_, columns=X.columns)

# Add row names to the DataFrame
df_pca_components.index = ['PC{}'.format(
    i+1) for i in range(len(df_pca_components.index))]

# # Print the principal components and their weights
# print("Principal components and their weights:\n", df_pca_components)

# Calculate the cumulative explained variance ratio
cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)

# Create the scree plot
plt.plot(range(1, len(cumulative_var_ratio)+1), cumulative_var_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
