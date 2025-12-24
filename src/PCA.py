import numpy as np

class PCA:
    def __init__(self, number_of_components):
        self.number_of_components = number_of_components
        # those attributes will be set when fitting the model
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None
        self.explained_variance_ratio = None
        
    # fit function calculates data's mean, eigenvalues, and eigenvectors
    # saves the data as class attributes to be used in transform function
    def fit(self, X):
        
        # Z = X - mean
        self.mean = np.mean(X, axis=0) # axis=0 gets the mean of each feature
        X_centered = X - self.mean     # centered data Z
        
        # covariance matrix (Sigma)
        covariance_matrix = np.cov(X_centered, rowvar=False) # rowvar=False to treat columns as numbers
        
        # eigenvalues and eigenvectors
        # solve for Sigma * v = lambda * v
        # where v are the eigenvectors and lambda are the eigenvalues
        # eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
         
        # sort eigenvalues and eigenvectors in descending order, since we need the highest ones for PCA
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # sort them using the right indices
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        
        if self.number_of_components <= self.eigenvectors.shape[1]:
            # use the top 'number_of_components' eigenvectors as principal components
            self.components = self.eigenvectors[:, :self.number_of_components] # top eigenvectors
            self.eigenvalues = self.eigenvalues[:self.number_of_components] # top eigenvalues
        else:
            raise ValueError("Number of components cannot be greater than the number of features")
        
        # total variance = sum of all eigenvalues used for principal components
        total_variance = np.sum(self.eigenvalues)
        
        # explained variance ratio = eigenvalue / total variance (array)
        # this number indicates how much variance is shown by the selected components
        self.explained_variance_ratio = self.eigenvalues / total_variance 