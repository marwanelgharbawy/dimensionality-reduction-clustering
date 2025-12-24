import numpy as np

class KMeans:
    def __init__(self, k=3, tolerance=1e-3, max_iterations=1000, initialization_method='random'):
        # number of clusters
        self.k = k
        self.initialization_method = initialization_method
            
        # convergence criteria
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # attributes to be set during fitting
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        if self.initialization_method == 'random':
            self.centroids = self.initialize_random_centroids(X)
        elif self.initialization_method == 'k-means++':
            self.centroids = self.initialize_kmeans_plus_plus(X)
        else:
            raise ValueError("Invalid initialization method")
        
    def initialize_random_centroids(self, X):
        n_samples = X.shape[0]
        # choose random k unique samples as initial centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[random_indices]
        return centroids
    
    def initialize_kmeans_plus_plus(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # randomly choose the first centroid
        first_centroid_index = np.random.randint(0, n_samples)
        centroids[0] = X[first_centroid_index]
        
        for i in range(1, self.k):
            # compute d^2 from the nearest centroid
            distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
            distances_squared = distances ** 2
            
            # get weighted probabilities
            probabilities = distances_squared / np.sum(distances_squared)
            
            # choose the next centroid based on the computed probabilities
            next_centroid_index = np.random.choice(n_samples, p=probabilities)
            
            centroids[i] = X[next_centroid_index] # next centroid
        
        return centroids