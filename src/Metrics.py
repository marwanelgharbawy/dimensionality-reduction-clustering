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

# silhouette score measures how similar an object is to its own cluster and how far away it is from other clusters
# score (b - a) / max(a, b)
def silhouette_score_scratch(X, labels):
    n_samples = X.shape[0]
    
    # get the k labels
    unique_labels = np.unique(labels)
    
    # instead of looping through each sample to get the average distances of points with and without
    
    # shape: (N, 1, D)
    A = X[:, np.newaxis, :] 
    
    # shape: (1, N, D)
    B = X[np.newaxis, :, :] 
    
    # subtract A - B -> automatically expanded to match dimensions.
    # result shape: (N, N, D) -> A 3D cube of differences.
    diff = A - B
    
    # this matrix contains pairs of distances between all points
    distance_matrix = np.linalg.norm(diff, axis=2) # shape: (N, N)
    
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        
        # calculate a[i] (tightness)
        # the mean distance from point i to all other points within the same cluster
        
        own_cluster = labels[i] # cluster label of point i
        
        # find indices within the same cluster
        indices_in_own_cluster = np.where(labels == own_cluster)[0]
        
        if len(indices_in_own_cluster) > 1:
            # get row i from the big distance matrix but only the columns of same cluster
            distances_within_cluster = distance_matrix[i, indices_in_own_cluster]
            
            # Sum them up and divide by count of points within cluster - 1 (-1 is the point itself)
            a_i = np.sum(distances_within_cluster) / (len(indices_in_own_cluster) - 1)
        else:
            a_i = 0.0
            
        # calculate b[i] (separation)   
        # the mean distance from point i to the nearest neighboring cluster
        
        b_i = np.inf # initialize to a large value
        
        # loop over other clusters (skipping self)
        for label in unique_labels:
            # skip own cluster
            if label == own_cluster:
                continue
                
            # Get indices of points in this "other" cluster
            indices_other = np.where(labels == label)[0]
            
            if len(indices_other) > 0:
                # Calculate mean distance to ALL points in this foreign cluster
                avg_dist_other = np.mean(distance_matrix[i, indices_other])
                
                # get nearest cluster -> keep the smallest average distance found so far
                if avg_dist_other < b_i:
                    b_i = avg_dist_other
        
        # Silhouette score for point: s[i] = (b - a) / max(a, b)
        
        max_ab = np.maximum(a_i, b_i)
        
        if max_ab == 0:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b_i - a_i) / max_ab
            
    # average score of all points
    return np.mean(silhouette_values)
    
# gap statistic to determine optimal k
# run kmeans on fake reference data (within the same bounding box) and compare inertia with real data
# for each k, the highest gap indicates the best k
def calculate_gap_statistic(X, k_values, kmeans_class, n_refs=5):
    gaps = []
    std_diffs = []
    
    # generate noise within the same min/max limits as the features
    mins = np.min(X, axis=0) # shape : (n_features,1)
    maxs = np.max(X, axis=0) # shape : (n_features,1)
    
    # loop over each k (cluster count) to calculate gap statistic
    for k in k_values:
        # run KMeans on data
        km = kmeans_class(k=k, max_iterations=100, initialization_method='k-means++')
        km.fit(X)
        
        # use the final inertia from history
        ref_inertia = km.inertia_history[-1]
            
        log_inertia_real = np.log(ref_inertia + 1e-10) # 1e-10 to avoid log(0)
        
        # run KMeans on reference data multiple times
        reference_log_inertias = []
        
        for i in range(n_refs):
            # generate random uniform data within min/max bounds
            X_ref = np.random.uniform(mins, maxs, X.shape)
            
            # run KMeans on this reference fake data
            km_ref = kmeans_class(k=k, max_iterations=100, initialization_method='k-means++')
            km_ref.fit(X_ref)
            
            inertia_ref = km_ref.inertia_history[-1]
                
            reference_log_inertias.append(np.log(inertia_ref + 1e-10))
            
        # calculate gap
        # formula: gap = E[log(W_ref)] - log(W_real)
        mean_log_ref = np.mean(reference_log_inertias)
        gap = mean_log_ref - log_inertia_real
        gaps.append(gap)
        
        # Calculate standard deviation (for the selection rule)
        sd_k = np.std(reference_log_inertias)
        s_k = sd_k * np.sqrt(1 + 1/n_refs)
        std_diffs.append(s_k)
        
        print(f"  k={k}: Gap={gap:.4f}")
        
    return gaps, std_diffs

def wcss_score(X, labels):
    wcss = 0
    unique_labels = np.unique(labels)
    
    for k in unique_labels:
        # get points belonging to this cluster
        cluster_points = X[labels == k]
        
        # get centroid
        centroid = np.mean(cluster_points, axis=0)
        
        # calculate squared euclidean distances and sum them up
        squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
        
        wcss += np.sum(squared_distances)
    return wcss