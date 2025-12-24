import numpy as np

def davies_bouldin_score(X, labels):
    # Get unique labels (clusters) ignoring noise if any (-1)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # If less than 2 clusters, score is not defined (or infinity)
    if n_clusters < 2:
        return np.inf
    
    #Compute Centroids and Intra-cluster Scatter (s_i)
    centroids = []
    s_values = [] # s_i
    
    for k in unique_labels:
        # Get points belonging to cluster k
        cluster_points = X[labels == k]
        
        # Centroid of cluster k
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        
        # Average Euclidean distance from points to centroid
        distances = np.sqrt(np.sum((cluster_points - centroid)**2, axis=1))
        s_i = np.mean(distances)
        s_values.append(s_i)
        
    centroids = np.array(centroids)
    s_values = np.array(s_values)
    
    # Compute Ratio R_ij and find max for each cluster
    db_score = 0
    
    for i in range(n_clusters):
        max_ratio = -np.inf
        
        for j in range(n_clusters):
            if i == j:
                continue
            
            # Distance between centroids d_ij
            d_ij = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
            
            # Avoid division by zero
            if d_ij == 0:
                # If centroids overlap completely so it's bad (max similarity)
                ratio = np.inf 
            else:
                ratio = (s_values[i] + s_values[j]) / d_ij
                
            if ratio > max_ratio:
                max_ratio = ratio
                
        db_score += max_ratio
        
    # Average over all clusters
    return db_score / n_clusters 




def calinski_harabasz_score_scratch(X, labels):
    
    
    # X: np.array of shape (n_samples, n_features)
    # labels: np.array of shape (n_samples,) containing cluster assignments
    
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    # If only one cluster or clusters equal to number of samples, index is undefined
    if k <= 1 or k >= n_samples:
        return 0.0

    # Calculate the global centroid (mean of all data points)
    global_centroid = np.mean(X, axis=0)

    extra_cluster_dispersion = 0.0
    intra_cluster_dispersion = 0.0

    for label in unique_labels:
        # Get data points belonging to the current cluster
        cluster_points = X[labels == label]
        cluster_centroid = np.mean(cluster_points, axis=0)
        n_q = len(cluster_points)

        #  Between-cluster dispersion (SSB)
        # Weight the distance between cluster centroid and global centroid by cluster size
        dist_to_global = np.sum((cluster_centroid - global_centroid) ** 2)
        extra_cluster_dispersion += n_q * dist_to_global

        # Within-cluster dispersion (SSW)
        # Sum of squared distances from points to their own cluster centroid
        intra_cluster_dispersion += np.sum((cluster_points - cluster_centroid) ** 2)

    # Apply the CH Formula: (SSB / (k - 1)) / (SSW / (n_samples - k))
    score = (extra_cluster_dispersion / (k - 1)) / (intra_cluster_dispersion / (n_samples - k))
    
    return score

