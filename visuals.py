import os
import argparse
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

# read data using pandas
data = pd.read_csv("./data/data.csv")
genres_data = pd.read_csv("./data/data_by_genres.csv")
year_data = pd.read_csv("./data/data_by_year.csv")


# visual showing Feature Correlation with Popularity
from yellowbrick.target import FeatureCorrelation

def featureCorrelationVisualizer():
    """
    Function which displays a visual feature correlation between all other features and popularity of music.

    Other features in dataset include: acousticness, danceability, energy, instrumentalness, liveness, etc.

    1 indicates a perfect positive linear relationship,
    -1 indicates a perfect negative linear relationship, and
    0 indicates no linear relationship.

    """
    other_features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

    X, y = data[other_features], data['popularity']

    # Create a list of the feature names
    features = np.array(other_features)

    # Instantiate the visualizer
    visualizer = FeatureCorrelation(labels=features, title="Feature Correlation with Popularity")

    plt.rcParams['figure.figsize']=(20,20)
    visualizer.fit(X, y)     # Fit the data to the visualizer
    visualizer.show()

def musicTrendVisualizer():
    """
    Function which displays trends in music features over the years of 1920 to 2020.
    """
    sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
    fig = px.line(year_data, x='year', y=sound_features)
    fig.show()

def genreFeatureVisualizer():
    top_genres = genres_data.nlargest(5, 'popularity')

    fig = px.bar(top_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
    fig.show()

def genreClusterVisualizer():

    # Pipeline to standardize data & then apply the kmeans clustering algorithm to it.
    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])

    # using genres_data to get strictly genre numerical data
    X = genres_data.select_dtypes(np.number)
    cluster_pipeline.fit(X)

    # add column to genres_data data frame which contains the cluster which the genre will belong to
    genres_data['cluster'] = cluster_pipeline.predict(X)

    # visualize data with tSNE
    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genres_data['genres']
    projection['cluster'] = genres_data['cluster']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    fig.show()

def main():
    parser = argparse.ArgumentParser(prog="visuals", description="visuals of music data set")

    visArray = ["featureCorrelation", "musicTrend", "genreFeatures", "genreClusters"]
    parser.add_argument('visualize', choices=visArray, help="Pick a way to visualize the data!")

    args = parser.parse_args()

    if (args.visualize == "featureCorrelation"):
       featureCorrelationVisualizer()
    elif (args.visualize == "musicTrend"):
       musicTrendVisualizer()
    elif (args.visualize == "genreFeatures"):
       genreFeatureVisualizer()
    elif (args.visualize == "genreClusters"):
        genreClusterVisualizer()

if __name__ == "__main__":
    main()


