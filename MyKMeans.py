import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random

class KMeansClustering(object):
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidian_distance(data_point, centroids):
        #computez distanta dintre un punct de data si toti centroizi
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        #initializez centroizi random dar neaparat in spatiul unde s-ar putea afla valori
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), 
                                           size=(self.k, X.shape[1]))

        #creez label pentru clustere
        for _ in range(max_iterations):
            y = []
            #computez distanta dintre fiecare punct de data si centroid
            for data_point in X:
                #lista pentru distantele dintre punctul curent si toate distantele
                distances = KMeansClustering.euclidian_distance(data_point, self.centroids)
                #aflu care centroid are cea mai mica distanta pana la punctul curent    
                cluster_num = np.argmin(distances) #acesta va fi indexul lui
                y.append(cluster_num)

            y = np.array(y)

            #Incep ajustez pozitia centroizilor 
            cluster_indices = []
            for i in range(self.k): #pentru fiecare cluster
                cluster_indices.append(np.argwhere(y == i)) #atribui fiecarui cluster indicele corespunzator

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    #daca nu are puncte de data care sa apartina acestui grup nu mut centroidul
                    cluster_centers.append(self.centroids[i]) 
                else:
                    #pozitionez centroidul la mijlocul distantei
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            cluster_centers = np.array(cluster_centers)

            #daca centroizi nu isi schimba pozitia mai mult decat valoarea precizata atunci se indeplineste conditia de oprire
            if np.max(np.linalg.norm(self.centroids - cluster_centers, axis=1)) < 0.001:
                break
            else: 
                self.centroids = cluster_centers

            return y 

    def make_data(self):
        data = make_blobs(n_samples=100, n_features=2, centers=random.randint(2,10))
        random_points = data[0]
        return random_points

    def make_plot(self, random_points, clusters):
        kmeans = KMeansClustering(k = clusters)
        labels = kmeans.fit(random_points)

        plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
                    marker='*', s=200)

        plt.show()

    def make_metrics(self, X):
        inertia = []
        silhuuette_scores = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
            #fac scorul silhouette
            if k > 1:
                silhuuette_avg = silhouette_score(X, kmeans.labels_)
                silhuuette_scores.append(silhuuette_avg)
        
        #creez graficul pentru metoda elbow
        #creez graficul pentru inertia
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), inertia, marker = 'o')
        plt.title('Metoda Elbow')
        plt.xlabel('Numarul de clustere')
        plt.ylabel('Inertia')
        plt.show()
        #creez graficul pentru silhouette score
        plt.figure(figsize=(8, 6))
        plt.plot(range(2,11), silhuuette_scores, marker= 'o')
        plt.title('Metoda Elbow')
        plt.xlabel('Numarul de clustere')
        plt.ylabel('Scorul Silhouette')
        plt.show()

    
