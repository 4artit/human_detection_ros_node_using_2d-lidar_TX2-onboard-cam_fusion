from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np

kmeans = KMeans(n_clusters=2, random_state = 0, n_init=3, max_iter=15)
dbs = DBSCAN(eps=0.5, min_samples=4)

def get_center_from_KNN(valid_points):
    kmeans.fit(np.asarray(valid_points))
    x = kmeans.cluster_centers_[:,0]
    y = kmeans.cluster_centers_[:,1]
    dist = np.sqrt(x**2 + y**2)
    return dist, x, y

def get_center_from_DBSCAN(valid_points):
    dbs.fit(np.asarray(valid_points))
    center = np.mean(dbs.components_, axis=0)
    x = center[0]
    y = center[1]
    dist = np.sqrt(x**2 + y**2)
    ang = np.arctan2(y, -x) * (180 / np.pi)
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


