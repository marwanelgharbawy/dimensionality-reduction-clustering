import numpy as np

def contingency_matrix(y_true, y_pred):
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    
    # Initialize matrix
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    
    # Populate matrix
    for i in range(n_classes):
        for j in range(n_clusters):
            # Boolean indexing to find intersection
            # Common elements in Class i AND Cluster j
            contingency[i, j] = np.sum((class_idx == i) & (cluster_idx == j))
            
    return contingency

def purity_score(y_true, y_pred):
    # Compute contingency matrix
    contingency = contingency_matrix(y_true, y_pred)
    
    # For each cluster (column), find the max number of samples from a single class
    # Sum these maximums and divide by total number of samples
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def adjusted_rand_index(y_true, y_pred):
    n_samples = y_true.shape[0]
    contingency = contingency_matrix(y_true, y_pred)
    
    # Calculate sums of combinations
    # Binomial coefficient "n choose 2" = n(n-1)/2
    def comb2(n):
        return n * (n - 1) / 2
    
    # Sum over rows (a_i) and columns (b_j)
    sum_comb_c = np.sum(comb2(contingency)) # Sum of binom(n_ij, 2)
    sum_comb_a = np.sum(comb2(np.sum(contingency, axis=1))) # Sum of binom(a_i, 2)
    sum_comb_b = np.sum(comb2(np.sum(contingency, axis=0))) # Sum of binom(b_j, 2)
    
    # Total combinations
    comb_n = comb2(n_samples)
    
    # Expected Index (under random model)
    expected_index = (sum_comb_a * sum_comb_b) / comb_n
    
    # Max Index
    max_index = (sum_comb_a + sum_comb_b) / 2
    
    # Handle denominator is 0 case
    if max_index == expected_index:
        return 1.0 if sum_comb_c == expected_index else 0.0
        
    return (sum_comb_c - expected_index) / (max_index - expected_index)

def normalized_mutual_information(y_true, y_pred):
    n_samples = y_true.shape[0]
    contingency = contingency_matrix(y_true, y_pred)
    
    # Marginal probabilities
    # p(y) = row_sums / N
    # p(c) = col_sums / N
    pi = np.sum(contingency, axis=1) / n_samples
    pj = np.sum(contingency, axis=0) / n_samples
    
    # Joint probabilities p(y, c)
    pij = contingency / n_samples
    
    # Entropy H(Y) and H(C)
    # H = -sum(p * log(p))
    # Add epsilon or mask to avoid log(0)
    pi = pi[pi > 0]
    pj = pj[pj > 0]
    
    h_true = -np.sum(pi * np.log(pi))
    h_pred = -np.sum(pj * np.log(pj))
    
    # Mutual Information I(Y; C)
    # I = sum(p_ij * log(p_ij / (p_i * p_j)))
    mi = 0
    
    # Iterate only where contingency > 0 to avoid log(0)
    rows, cols = np.nonzero(contingency)
    for r, c in zip(rows, cols):
        p_joint = pij[r, c]
        p_prod = (np.sum(contingency, axis=1)[r] / n_samples) * \
                 (np.sum(contingency, axis=0)[c] / n_samples)
        mi += p_joint * np.log(p_joint / p_prod)
        
    # NMI
    denominator = h_true + h_pred
    
    if denominator == 0:
        return 0.0
        
    return 2 * mi / denominator