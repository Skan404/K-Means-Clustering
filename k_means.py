import numpy as np

def initialize_centroids_forgy(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def initialize_centroids_kmeans_pp(data, k):
    centroids = [data[np.random.choice(range(data.shape[0]))]]
    for _ in range(1, k):
        distances = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in data])
        new_centroid_index = np.argmax(distances)
        centroids.append(data[new_centroid_index])
    return np.array(centroids)

def assign_to_cluster(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, assignments):
    new_centroids = np.array([data[assignments==i].mean(axis=0) for i in range(assignments.max()+1)])
    return new_centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100):
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

