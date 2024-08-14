import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Read in the data file containing your audio features for each song in each city and week
data = pd.read_csv('clean-std-data.csv')
# Extract the audio features from the dataset
X = data.iloc[:, 1:]

# Create a pipeline for PCA
pipe = Pipeline([
    ('pca', PCA()),
])

# Define the parameter grid for n_components
param_grid = {
    'pca__n_components': range(1, X.shape[1]+1),
}

# Perform a grid search over n_components using cross-validation
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X)

# Print the results of the grid search
print("Best number of components:", grid.best_params_['pca__n_components'])
print("Explained variance ratio:", grid.best_score_)
