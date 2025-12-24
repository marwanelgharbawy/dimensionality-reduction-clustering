import numpy as np

class KMeans:
    def __init__(self, k=3, tolerance=1e-3, max_iterations=1000, initialization_method='k-means++'):
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
        # initialize centroids
        if self.initialization_method == 'k-means++':
            self.centroids = self.initialize_kmeans_plus_plus(X)
        elif self.initialization_method == 'random':
            self.centroids = self.initialize_random_centroids(X)
        else:
            raise ValueError("Invalid initialization method")
        
        # assign, then update
        for iteration in range(self.max_iterations):
            # assign clusters based on current centroids
            self.labels = self.assign_clusters(X) # indices
            
            # store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # update centroids based on current assignments
            new_centroids = np.zeros_like(self.centroids)
            
            for k in range(self.k):
                # get all points assigned to cluster k
                cluster_points = X[self.labels == k]
                
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # handle empty cluster if no points are assigned
                    new_centroids[k] = self.centroids[k]
                    
            self.centroids = new_centroids
            
            # check for convergence
            centroids_distance = np.linalg.norm(self.centroids - old_centroids)
            if centroids_distance < self.tolerance:
                print(f"Converged at iteration number {iteration + 1}.")
                return # no more iterations
        
        print(f"Converged after reaching maximum iterations ({self.max_iterations}).")
        
        
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
            distances = self._calculate_distances(X, centroids[:i]) # shape: (n_samples, i)
    
            # get the distance to the nearest centroid for each sample
            min_distances = np.min(distances, axis=1) # shape: (n_samples, 1)
            
            distances_squared = min_distances ** 2
            
            # get weighted probabilities
            probabilities = distances_squared / np.sum(distances_squared)
            
            # choose the next centroid based on the computed probabilities
            next_centroid_index = np.random.choice(n_samples, p=probabilities)
            
            centroids[i] = X[next_centroid_index] # next centroid

        return centroids
            
    def assign_clusters(self, X):
        distances = self._calculate_distances(X, self.centroids)
        
        # return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1) # shape: (n_samples, 1) representing the k clusters they belong to
        
    def _calculate_distances(self, X, centroids):
        distances_list = [np.linalg.norm(X - centroid, axis=1) for centroid in centroids]
        return np.array(distances_list).T  # shape: (n_samples, n_centroids)
    
    def predict(self, X):
        return self.assign_clusters(X)