import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from MyKMeans import KMeansClustering
from HierarhicalClustering import hierarficalClustering

#incarc setul de date
file_path = "Country-data.csv"
data = pd.read_csv(file_path)

#print(data)

#encodez atributul de country
if 'country' in data.columns:
    encoder = LabelEncoder()
    data['country'] = encoder.fit_transform(data['country'])

#scalez datele
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

#utilizez metoda elbow
inertia = []
silhuuette_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)
    #fac scorul silhouette
    if k > 1:
        silhuuette_avg = silhouette_score(data_scaled, kmeans.labels_)
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

#alegem numarul de clustere si se desfasoara metoda k-means
optimal_k = int(input("Ce numar de clustere doriti? "))
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

#adaug etichete de clustere pentru datele originale
data['Cluster'] = clusters

#creez graficul pentru exporturi si inflatie
plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=data['exports'],
    y=data['inflation'],
    hue=data['Cluster'],
    palette='viridis',
    s=100
    )

#pun titlurile pe grafic si pe axe
plt.title("Clustere bazate pe exporturi si inflatie", fontsize=16)
plt.xlabel("Exporturi", fontsize=12)
plt.ylabel("Inflatie", fontsize=12)
plt.legend(title="Clustere")
plt.show()

plt.figure(figsize=(10, 7))
#creez graficul pentru importuri si venit
sns.scatterplot(
    x=data['imports'],
    y=data['income'],
    hue=data['Cluster'],
    palette='viridis',
    s=100
    )

#pun titlurile pe grafic si pe axe
plt.title("Clustere bazate pe importuri si venit", fontsize=16)
plt.xlabel("Importuri", fontsize=12)
plt.ylabel("Venit", fontsize=12)
plt.legend(title="Clustere")
plt.show()

demo = input("Doriti sa vizualizati graficul 2D? y/n")
if demo == 'y':
    km = KMeansClustering()
    data = km.make_data()
    km.make_metrics(data)
    cl = int(input('Cate clustere doriti?'))
    km.make_plot(data, clusters=cl)
    km = KMeansClustering()
    data = km.make_data()
    km.make_metrics(data)
    cl = int(input('Cate clustere doriti?'))
    km.make_plot(data, clusters=cl)
    km = KMeansClustering()
    data = km.make_data()
    km.make_metrics(data)
    cl = int(input('Cate clustere doriti?'))
    km.make_plot(data, clusters=cl)
    
else:
    print('Nu ati dorit sa vizualizati')

hc = hierarficalClustering()
hc.fit_hierarhicalClustering()

