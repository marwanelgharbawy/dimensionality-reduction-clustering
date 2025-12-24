import numpy as np

class PCA:
    def __init__(self, number_of_components):
        self.number_of_components = number_of_components
        # those attributes will be set when fitting the model
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None      # principal components, also the compression matrix
        self.top_eigenvalues = None # eigenvalues for the selected components
        self.explained_variance_ratio = None
        
    # fit function calculates data's mean, eigenvalues, and eigenvectors
    # saves the data as class attributes to be used in transform function
    def fit(self, X):
        
        # Z = X - mean
        self.mean = np.mean(X, axis=0) # axis=0 gets the mean of each feature
        X_centered = X - self.mean     # centered data Z
        
        # covariance matrix (Sigma)
        # size is [n_features x n_features] (in our case, 30 x 30) 
        # shows how features vary with respect to each other (symmetric matrix)
        covariance_matrix = np.cov(X_centered, rowvar=False) # rowvar=False to treat columns as numbers
        
        # eigenvalues and eigenvectors
        # solve for Sigma * v = lambda * v
        # where v are the eigenvectors and lambda are the eigenvalues
        # eigh returns eigenvalues in ascending order
        # in our case, we get 30 eigenvalues and 30 eigenvectors (one for each feature)
        # best eigenvectors to use for PCA are the ones with highest eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
         
        # sort eigenvalues and eigenvectors in descending order, since we need the highest ones for PCA
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # sort them using the right indices
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        
        if self.number_of_components > self.eigenvectors.shape[1]:
            raise ValueError("Number of components cannot be greater than the number of features")
            
        # use the top 'number_of_components' eigenvectors as principal components using slicing
        self.components = self.eigenvectors[:, :self.number_of_components]
        self.top_eigenvalues = self.eigenvalues[:self.number_of_components]
        
        # total variance = sum of all eigenvalues used for principal components
        total_variance = np.sum(self.eigenvalues)
        
        # explained variance ratio = eigenvalue / total variance (array)
        # this number indicates how much variance is shown by the selected components
        self.explained_variance_ratio = self.top_eigenvalues / total_variance 
    
    # apply the dimensionality reduction on data X
    # X_new = Z . W
    # where Z is centered data and W is the compression matrix (principal components)
    # called right after fit
    def transform(self, X):
        Z = X - self.mean
        return np.dot(Z, self.components) # reduced data
    
    # reconstruct data back to original space
    # X_reconstructed = X_reduced . W^T + mean
    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components.T) + self.mean
    
    # calculate the reconstruction error (MSE) between original and reconstructed data
    def get_reconstruction_error(self, X):
        # transform and inverse transform the data with PCA
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        return np.mean((X - X_reconstructed) ** 2)