import numpy as np

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, cov_type='full', reg_covar=1e-6, random_state=42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.cov_type = cov_type
        self.reg_covar = reg_covar
        self.random_state = random_state

        # Parameters to be learned
        self.means = None
        self.covariances = None
        self.priors = None
        self.log_likelihood_history = []
        self.converged_ = False

    def _initialize_parameters(self, X):
        rng = np.random.default_rng(self.random_state) #Random initialization of parameters
        n_samples, n_features = X.shape

        # Initialize Means as randomly select n components points from X
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]

        # Initialize Priors as Uniformly distributed
        self.priors = np.full(self.n_components, 1 / self.n_components)

        # Initialize Covariances based on type
        if self.cov_type == 'full':
            # Shape: (K, D, D) - Identity matrices
            self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        elif self.cov_type == 'tied':
            # Shape: (D, D) - Single Identity matrix
            self.covariances = np.eye(n_features)
        elif self.cov_type == 'diagonal':
            # Shape: (K, D) - Ones
            self.covariances = np.ones((self.n_components, n_features))
        elif self.cov_type == 'spherical':
            # Shape: (K,) - Ones
            self.covariances = np.ones(self.n_components)

    def _log_pdf_multivariate_gaussian(self, X, mean, cov):
        n_samples, n_features = X.shape
        X_centered = X - mean

        if self.cov_type == 'full':
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                # Fallback if matrix is not positively definite (add more regularization)
                cov = cov + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov)

            log_det_cov = 2 * np.sum(np.log(np.diagonal(L)))
            Y = np.linalg.solve(L, X_centered.T)
            mahalanobis = np.sum(Y**2, axis=0)

        elif self.cov_type == 'tied':
            # Same logic as full but cov is shared
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                cov = cov + np.eye(n_features) * self.reg_covar
                L = np.linalg.cholesky(cov)
            log_det_cov = 2 * np.sum(np.log(np.diagonal(L)))
            Y = np.linalg.solve(L, X_centered.T)
            mahalanobis = np.sum(Y**2, axis=0)

        elif self.cov_type == 'diagonal':
            # cov is shape (D,). determinant is product of diagonal elements
            log_det_cov = np.sum(np.log(cov))
            mahalanobis = np.sum((X_centered**2) / cov, axis=1)

        elif self.cov_type == 'spherical':
            # cov is scalar sigma^2.
            log_det_cov = n_features * np.log(cov)
            mahalanobis = np.sum(X_centered**2, axis=1) / cov

        const = n_features * np.log(2 * np.pi)
        return -0.5 * (const + log_det_cov + mahalanobis)

    def _e_step(self, X):
        n_samples = X.shape[0]
        weighted_log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            if self.cov_type == 'full':
                cov = self.covariances[k]
            elif self.cov_type == 'tied':
                cov = self.covariances
            elif self.cov_type == 'diagonal':
                cov = self.covariances[k]
            elif self.cov_type == 'spherical':
                cov = self.covariances[k]


            weighted_log_prob[:, k] = np.log(self.priors[k]) + \
                                      self._log_pdf_multivariate_gaussian(X, self.means[k], cov)

        max_log_prob = np.max(weighted_log_prob, axis=1, keepdims=True)
        log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob - max_log_prob), axis=1, keepdims=True)) + max_log_prob

        log_resp = weighted_log_prob - log_prob_norm

        # Return actual responsibilities and the mean log-likelihood
        return np.exp(log_resp), np.mean(log_prob_norm)

    def _m_step(self, X, resp):

        n_samples, n_features = X.shape

        # Update total weight per cluster (N_k)
        Nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps # Add epsilon to avoid division by zero

        # Update Priors
        self.priors = Nk / n_samples

        # Update Means
        self.means = np.dot(resp.T, X) / Nk[:, np.newaxis]

        # Update Covariances
        if self.cov_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means[k] # (N, D)

                weighted_diff = resp[:, k][:, np.newaxis] * diff
                cov_k = np.dot(weighted_diff.T, diff) / Nk[k]
                # Regularize
                cov_k.flat[::n_features + 1] += self.reg_covar
                self.covariances[k] = cov_k

        elif self.cov_type == 'tied':
            # Average over all clusters
            avg_cov = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                weighted_diff = resp[:, k][:, np.newaxis] * diff
                avg_cov += np.dot(weighted_diff.T, diff)
            avg_cov /= n_samples # Divided by total N because it's tied
            avg_cov.flat[::n_features + 1] += self.reg_covar
            self.covariances = avg_cov

        elif self.cov_type == 'diagonal':

            self.covariances = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff_sq = (X - self.means[k]) ** 2
                self.covariances[k] = np.sum(resp[:, k][:, np.newaxis] * diff_sq, axis=0) / Nk[k]
            self.covariances += self.reg_covar

        elif self.cov_type == 'spherical':
            # Average variance across all dimensions
            self.covariances = np.zeros(self.n_components)
            for k in range(self.n_components):
                diff_sq = (X - self.means[k]) ** 2 # (N, D)
                # Sum over samples & features
                term = np.sum(resp[:, k][:, np.newaxis] * diff_sq)
                self.covariances[k] = term / (Nk[k] * n_features)
            self.covariances += self.reg_covar

    def fit(self, X):
        # EM
        self._initialize_parameters(X)
        self.log_likelihood_history = []

        for i in range(self.max_iter):
            prev_log_likelihood = self.log_likelihood_history[-1] if self.log_likelihood_history else -np.inf

            # E-step
            resp, log_likelihood = self._e_step(X)
            self.log_likelihood_history.append(log_likelihood)

            # Check convergence
            if i > 0 and abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged_ = True
                print(f"Converged at iteration {i}")
                break

            # M-step
            self._m_step(X, resp)

        return self

    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)

    def bic(self, X):
        _, log_likelihood_mean = self._e_step(X)
        log_likelihood_sum = log_likelihood_mean * X.shape[0]
        n_features = X.shape[1]

        # Count parameters
        if self.cov_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.cov_type == 'diagonal':
            cov_params = self.n_components * n_features
        elif self.cov_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        elif self.cov_type == 'spherical':
            cov_params = self.n_components

        n_params = (self.n_components * n_features) + cov_params + (self.n_components - 1)
        return -2 * log_likelihood_sum + n_params * np.log(X.shape[0])

    def aic(self, X):
        _, log_likelihood_mean = self._e_step(X)
        log_likelihood_sum = log_likelihood_mean * X.shape[0]
        n_features = X.shape[1]

        # Count parameters
        if self.cov_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.cov_type == 'diagonal':
            cov_params = self.n_components * n_features
        elif self.cov_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        elif self.cov_type == 'spherical':
            cov_params = self.n_components

        n_params = (self.n_components * n_features) + cov_params + (self.n_components - 1)
        return -2 * log_likelihood_sum + 2 * n_params