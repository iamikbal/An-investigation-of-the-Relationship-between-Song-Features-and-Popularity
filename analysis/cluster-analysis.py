import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
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

# Calculate linkage matrix from the distance matrix
# types of linkage => weighted single complete average centroid median ward
method_linkage = 'single'
linkage_matrix = hierarchy.linkage(
    distance_matrix_df, method=method_linkage, metric='euclidean')

# Define a list of custom labels for the x-axis
labels = ['Ahmedabad', 'Bengaluru', 'Chandigarh', 'Chennai', 'Delhi', 'Guwahati', 'Hyderabad',
          'Imphal', 'Jaipur', 'Kochi', 'Kolkata', 'Lucknow', 'Ludhiana', 'Mumbai', 'Patna', 'Pune']

# Plot dendrogram from the linkage matrix
dendrogram = hierarchy.dendrogram(
    linkage_matrix, labels=labels, leaf_rotation=90)
plt.xlabel(method_linkage, rotation='horizontal')

# Display the plot
plt.show()
