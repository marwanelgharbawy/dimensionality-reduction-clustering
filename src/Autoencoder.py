import numpy as np

class Autoencoder:
    def __init__ (self, layer_sizes, activations, l2_lambda = 0.01):
        self.layer_sizes = layer_sizes  #lists each layer size for example [input_size, hidden1_size, hidden2_size, ..., output_size]
        self.activations = activations  #list of activation functions for each transformation layer
        self.l2_lambda = l2_lambda  
        self.weights = []
        self.biases = []
        
        #intializing weights using He Init
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i])
            bias = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(weight)
            self.biases.append(bias)

    def _activation_function(self, x, function):
        if function == 'relu':
            return np.maximum(0, x)
        elif function == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clipping to avoid overflow
        elif function == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unsupported activation function")
        
    def _activation_derivative(self, x, function):
        if function == 'relu':
            return (x > 0).astype(float)
        elif function == 'sigmoid':
            return x * (1 - x)
        elif function == 'tanh':
            return 1 - np.power(x, 2)
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, X):
        X_T = X.T
        self.a = [X_T]
        self.z = []

        for i in range(len(self.weights)):
            Z = np.dot(self.weights[i], X_T) + self.biases[i]
            X_T = self._activation_function(Z,self.activations[i])
            self.z.append(Z)
            self.a.append(X_T)
        
        return X_T.T
    
    def backward(self, X, reconstructed_X, lr):
        m = X.shape[0]
        dA = reconstructed_X.T - X.T

        for i in reversed(range(len(self.weights))):
            dZ = dA * self._activation_derivative(self.a[i+1], self.activations[i])

            #gradients with l2 regulization
            dW = (1/m) * np.dot(dZ, self.a[i].T) + (self.l2_lambda/m) * self.weights[i]
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            #update dA for next (prev) layer
            dA = np.dot(self.weights[i].T, dZ)

            #update weights and biases
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    def train(self, X , epochs = 100 ,batch_size = 32, initial_lr = 0.1, decay = 0.01):
        loss_history = []
        for epoch in range(epochs):
            # learing rate step decay
            lr = initial_lr / (1 + decay * epoch)

            #shuffle for the mini batch gradiant decent
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X [indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                reconstructed_X = self.forward(X_batch)
                self.backward(X_batch, reconstructed_X, lr)

            #mean square loss
            reconstruction = self.forward(X)
            loss = np.mean(np.square(X - reconstruction))
            loss_history.append(loss)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {lr:.4f}')

        return loss_history

