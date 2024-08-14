import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('cluster01.csv')

# Standardize the data
scaled_df = StandardScaler().fit_transform(df.iloc[:, 3:])

# Apply PCA to reduce to 4 components
pca = PCA(n_components=4)
pca_components = pca.fit_transform(scaled_df)

# Convert the PCA components to a dataframe
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4'])

# Concatenate the original audio features with the PCA features
concatenated_df = pd.concat([pd.DataFrame(df.iloc[:, 3:]), pd.DataFrame(pca_df)], axis=1)

# Get the correlation matrix between PCA components and original variables
correlation_matrix = concatenated_df.corr()
# print(correlation_matrix)

# Save the correlation matrix to a CSV file
# pd.DataFrame(scaled_df).to_csv('std.csv', index=True)
correlation_matrix.to_csv('correlation_matrix01.csv', index=True)
