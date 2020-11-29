import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import style 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

import math
import datetime
data = pd.read_csv('FINAL.csv', encoding='utf-8')
sns.set(rc={'figure.figsize':(25,40)})
x = long = np.array(data['Longitude'])
y = lat = np.array(data['Latitude'])
X = np.vstack((long, lat)).T
sns_plot = sns.scatterplot(x=long,y=lat,data = data,s=5)
fig = sns_plot.get_figure()
fig.savefig("output.png")

# cost =[] 
# for i in range(1, 100): 
#     KM = KMeans(n_clusters = i, max_iter = 500) 
#     KM.fit(X) 
      
#     # calculates squared error 
#     # for the clustered points 
#     cost.append(KM.inertia_)      
  
# # plot the cost against K values 
# plt.plot(range(1, 11), cost, color ='g', linewidth ='3') 
# plt.xlabel("Value of K") 
# plt.ylabel("Sqaured Error (Cost)") 
# plt.show()





y_pred = KMeans(n_clusters=46).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, edgecolors='black', linewidths = 0.5, alpha = 0.8, s = 100)






