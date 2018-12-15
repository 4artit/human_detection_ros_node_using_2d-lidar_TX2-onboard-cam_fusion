from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

kmeans = KMeans(n_clusters=2, random_state = 0, n_init=3, max_iter=15)
dbs = DBSCAN(eps=1, min_samples=2)

def get_center_from_KNN(valid_points):
    kmeans.fit(np.asarray(valid_points))
    x = kmeans.cluster_centers_[:,0]
    y = kmeans.cluster_centers_[:,1]
    dist = np.sqrt(x**2 + y**2)
    return dist, x, y

def get_center_from_DBSCAN(valid_points):
    dbs.fit(np.asarray(valid_points))
    labels = dbs.labels_
    temp_x = np.array([0.0,0.0,0.0,0.0,0.0])
    temp_y = np.array([0.0,0.0,0.0,0.0,0.0])
    count = np.array([0.0,0.0,0.0,0.0,0.0])
    for i in range(len(labels)):
        if labels[i] >= 0:
            temp_x[labels[i]] = temp_x[labels[i]] + valid_points[i][0]
            temp_y[labels[i]] = temp_y[labels[i]] + valid_points[i][1]
            count[labels[i]] = count[labels[i]]+1.0
    temp_x = temp_x / count
    temp_y = temp_y / count
    min_dist = 9999
    min_index = -1
    for i in range(len(count)):
        if count[i] != 0:
            temp = temp_x[i] * temp_x[i] + temp_y[i] * temp_y[i]
            if min_dist > temp:
                min_dist = temp
                min_index = i
    dist = np.sqrt(min_dist)
    ang = np.arctan2(temp_y[min_index], -temp_x[min_index]) * (180 / np.pi)
    return dist, ang

def get_center_from_mid_points(valid_points):
    mid_index = len(valid_points)/2
    x = np.array([valid_points[mid_index-4][0], valid_points[mid_index-3][0], 
                                      valid_points[mid_index-2][0], valid_points[mid_index-1][0], valid_points[mid_index][0],
                                      valid_points[mid_index+1][0], valid_points[mid_index+2][0], valid_points[mid_index+3][0], valid_points[mid_index+4][0]])
    y = np.array([valid_points[mid_index-4][1], valid_points[mid_index-3][1],
                                      valid_points[mid_index-2][1], valid_points[mid_index-1][1], valid_points[mid_index][1],
                                      valid_points[mid_index+1][1], valid_points[mid_index+2][1], valid_points[mid_index+3][1], valid_points[mid_index+4][1]])
    dist = np.sqrt(x**2 + y**2)
    min_index = dist.argmin()
    ang = np.arctan2(y[min_index], -x[min_index]) * (180 / np.pi)
    return dist[min_index], ang


