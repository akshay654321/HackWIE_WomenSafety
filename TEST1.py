import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import style 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import json



data = pd.read_csv('FINAL.csv', encoding='utf-8')
sns.set(rc={'figure.figsize':(20,20)})
x = long = np.array(data['Longitude'])
y = lat = np.array(data['Latitude'])
xyz = np.vstack((lat, long)).T
X = np.vstack((long, lat)).T
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.1, min_samples=50).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

#print('Estimated number of clusters: %d' % n_clusters_)
#print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f"
#      % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f"
#      % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=15)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

centroids = []
for p in range(n_clusters_):
    peepee = []
    for q in range(len(labels)):
        if labels[q] == p:
            peepee.append(xyz[q])
    if p == 0:
        centroids.append(np.mean(peepee, axis=0))
        rename = np.mean(peepee, axis=0)
        centroids.append(np.mean(peepee, axis=0))
    else:
        centroids.append(np.mean(peepee, axis=0))
#    print(len(peepee))
    peepee = []


np.savetxt("foo.csv", centroids, delimiter=",")

cgeo = pd.read_csv('foo.csv', encoding='utf-8')
cgeo.columns = ['Latitude', 'Longitude']

def df_to_geojson(df, lat='latitude', lon='longitude'):
    geojson = {'type':'FeatureCollection', 'features':[]}
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                    'properties':{},
                    'geometry':{'type':'Point',
                                'coordinates':[]}}
        feature['geometry']['coordinates'] = [row[lon],row[lat]]

        geojson['features'].append(feature)
    return geojson



    

    
    
    

