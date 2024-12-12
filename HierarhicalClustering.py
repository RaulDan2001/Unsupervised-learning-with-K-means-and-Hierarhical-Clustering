import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

class hierarficalClustering(object):
    def __init__(self):
        self.data_scaled = None
        self.file_path = None
        self.data = None
        self.load_data()
        self.encode_scale_data()

    def load_data(self):
        self.file_path = "Country-data.csv"
        self.data = pd.read_csv(self.file_path)

    def encode_scale_data(self):
        if 'country' in self.data.columns:
            encoder = LabelEncoder()
            self.data['country'] = encoder.fit_transform(self.data['country'])

        scaler = StandardScaler()
        self.data_scaled = scaler.fit_transform(self.data)

    def fit_hierarhicalClustering(self):
        linked = linkage(self.data_scaled, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(
            linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=False,
            color_threshold=0)

        plt.title("Dendograma pentru Clusterele hierarhice")
        plt.xlabel("Probe")
        plt.ylabel("Distanta Euclidiana")
        plt.show()




