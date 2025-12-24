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