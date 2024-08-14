import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read in the data file containing your audio features for each song in each city and week
data = pd.read_csv('clean-std-data.csv')

# Select only the audio features
audio_features = data.iloc[:, 1:]

# # Remove duplicates
# data = data.drop_duplicates()
# # print(data)

# Remove highly correlated variables
# Calculate the correlation matrix
corr_matrix = audio_features.corr()
# Save the distance matrix DataFrame to a CSV file
# corr_matrix.to_csv('corr_matrix.csv', index=False)
# Print the correlation matrix
# print(corr_matrix)
high_corr_vars = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            if colname_i not in high_corr_vars:
                high_corr_vars.add(colname_j)
print(high_corr_vars)


# data = data.drop(columns=list(high_corr_vars))

# # Standardize the data
# audio_features = data.iloc[:, 2:]
# scaler = StandardScaler()
# audio_features_std = scaler.fit_transform(audio_features)


# Extract the audio features from the dataset
X = data.iloc[:, 1:]
# print(data)

# Perform PCA on the audio features
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Print the explained variance ratio for each principal component
print("Explained variance ratio:\n", pca.explained_variance_ratio_)

# Create a Pandas DataFrame for the principal components and their weights
df_pca_components = pd.DataFrame(pca.components_, columns=X.columns)

# Add row names to the DataFrame
df_pca_components.index = ['PC{}'.format(
    i+1) for i in range(len(df_pca_components.index))]

# Print the principal components and their weights
print("Principal components and their weights:\n", df_pca_components)

# # Save the pca_components DataFrame to a CSV file
# df_pca_components.to_csv('pca_components_weights.csv', index=False)