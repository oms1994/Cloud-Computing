from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

# setting fixed seed value 
np.random.seed(147)



def calc_dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)



def get_gauss_dist(m, dim, mean, stdev):
    covar_matrix = [[stdev**2, 0], [0, stdev**2]]
    x, y = np.random.multivariate_normal(mean=mean,cov=covar_matrix,size=m).T

    return [x, y]

def kmeans_clust(C, k, data, epochs):

    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(data.shape[0])
    
    # running kmeans_clust algorithm for n epochs
    for epoch in range(epochs):
        # Assigning each value to its closest cluster
        for i, instance in enumerate(data):
            distances = calc_dist(instance, C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
    
        # Storing old centroid values
        C_old = deepcopy(C)

        colors = ['lightgreen', 'c', 'lightpink', 'lightyellow', 'm']
        points = {}
        ax = fig.add_subplot(2, 5, epoch+1)
        for i in range(k):
            points[i] = np.vstack([data[j] for j in range(data.shape[0]) if clusters[j] == i])
            C[i] = np.mean(points[i], axis=0)
            ax.scatter(points[i][:, 0], points[i][:, 1], marker='p', color=colors[i], edgecolors='brown')
        ax.scatter(C_old[:, 0], C_old[:, 1], marker='o', s=150, color='r', edgecolors='brown')
        ax.set_title(label="After "+str(epoch+1)+" iterations")

## generate data points
data_values = {}
global fig
fig = plt.figure()

ax = fig.add_subplot(2,3,1)
centres = np.asarray([[3,5],[-5,2],[1,-4],[-5,-5],])
for i in range(0, centres.shape[0]):
    centre = centres[i]
    data_values[i+1] = get_gauss_dist(m=100, dim=2, mean=centre, stdev=4)

## preprocessing generated data points into one single array
x = np.zeros((1,))
y = np.zeros((1,))
for i in range(0, centres.shape[0]):
    x = np.hstack((x, data_values[i+1][0]))
    y = np.hstack((y, data_values[i+1][1]))
x = x[1:]
y = y[1:]
data = np.vstack((x,y)).T

## plot generated data
ax.scatter(x, y, marker='o', color='c', edgecolors='black')



C_x = np.random.randint(np.min(x), np.max(x), size=4)
C_y = np.random.randint(np.min(y), np.max(y), size=4)
ax.scatter(C_x, C_y, marker='*', color='r', s=150, edgecolors='black')


C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
kmeans_clust(C=C, k=4, data=data, epochs=10)

## display figure
plt.show(fig)