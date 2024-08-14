import pandas as pd
from sklearn.decomposition import PCA

# Read in the data file containing your audio features for each song in each city and week
data = pd.read_csv('cluster0.csv')

# Extract the audio features from the dataset
X = data.iloc[:, 3:]

# Perform PCA on the audio features
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Print the explained variance ratio for each principal component
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Create a Pandas DataFrame for the principal components and their weights
df_pca_components = pd.DataFrame(pca.components_, columns=X.columns)

# Add row names to the DataFrame
df_pca_components.index = ['PC{}'.format(
    i+1) for i in range(len(df_pca_components.index))]

# Print the principal components and their weights
print("Principal components and their weights:\n", df_pca_components)

# Save the PCA components weights DataFrame to a CSV file
df_pca_components.to_csv('pca_components_weights.csv', index=False)
