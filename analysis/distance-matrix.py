import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Read the data from a CSV file
data = pd.read_csv(
    'clean-std-data.csv')

# Calculate the pairwise distances between the rows
# Compute the Euclidean distance matrix
column = ["danceability", "energy", "loudness", "speechiness",
          "acousticness", "instrumentalness", "valence", "tempo"]
distances = pdist(data[column].values)

# Convert the distances to a square distance matrix
distance_matrix = squareform(distances)

# Convert the distance matrix to a pandas DataFrame
distance_matrix_df = pd.DataFrame(distance_matrix)

# Save the distance matrix DataFrame to a CSV file
distance_matrix_df.to_csv('distance_matrix.csv', index=False)
