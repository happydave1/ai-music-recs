import os
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

data = pd.read_csv("./data/data.csv")
genres_data = pd.read_csv("./data/data_by_genres.csv")
year_data = pd.read_csv("./data/data_by_year.csv")

print(data)
print(genres_data)
print(year_data)