import pandas as pd

# Read in the data file containing your audio features for each song in each city and week
data = pd.read_csv('cluster01.csv')

# Select only the audio features
audio_features = data.iloc[:, 3:]

# Calculate the correlation matrix
corr_matrix = audio_features.corr()
# Save the distance matrix DataFrame to a CSV file
corr_matrix.to_csv('corr_matrix5.csv', index=False)