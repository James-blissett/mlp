# Code from https://github.com/elcaiseri/Machine-Learning-from-Scratch?source=post_page-----e4cee82ab06d---------------------------------------
# The origional is for a regression task. I have turned it into a classification task. It was also a bit old.
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes          # each element in hidden_sizes represents a layer, magnutide is #neurons/layer
        self.output_size  = output_size
        self.num_layers   = len(hidden_sizes) + 1 # counts gaps between layers - each gap has corresponding weights and biases

        # Initialise the weights and biases for each layer (gaps b/w hidden layers, and also output layer)
        self.weights = []
        self.biases  = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(1, self.num_layers + 1):
            self.weights.append(np.random.randn(sizes[i], sizes[i-1]) * np.sqrt(2 / sizes[i-1])) # initialise weights as random numbers from standard normal distirbution, with He adjustment to make compatible with ReLU activation function
            self.biases.append(np.zeros((sizes[i],1)))                                           # initialise to 0, per He convention
    
    # forward pass
    def forward(self, X):
        # Forward pass through the network
        self.activations = [X]
        self.z = [] # pre-activation value
        
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                a = self.ReLU(z)            # ReLU activation function for hidden layers
            else:
                a = self.sigmoid(z)                   # linear activations for output layer
            self.activations.append(a) 
        return self.activations[-1]     # shape: (output_size, m)

    # backpropogation step
    def backprop(self, X, y):
        m = X.shape[1]  # Num input examples; The training dataset is parsed as X
    
        # Compute gradients
        gradients = []                  # initialise gradient list
        dZ = self.activations[-1] - y   # dL/dz, the y^{hat} - y step for the final layer

        for i in range(self.num_layers - 1, -1, -1):
            dW = (1 / m) * np.dot(dZ, self.activations[i].T)    # shape: (sizes[i-1], m) Equation 9
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)      # shape: (sizes[i-1], m) Equation 9
            gradients.append((dW, db))

            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)          # shape: (sizes[i-1], m)
                dZ = dA * self.ReLUgradient(self.z[i-1])   # shape: (sizes[i-1], m)
        
        return gradients[::-1]  # reverse the gradients
    
    # Gradient descent step
    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][0]

    # Activation function
    def ReLU(self, Z):
        return np.maximum(0, Z)
    
    # Activation function derivative
    def ReLUgradient(self, Z):
        return (Z > 0).astype(float)    # 1 where z > 0, else 0
        
    # Output function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

def plot_loss_curve(losses, save_path="loss_curve.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic classification dataset
    X, y = make_classification(n_samples = 100, n_features = 2, n_informative = 2, n_redundant = 0, random_state = 42)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

    # Normalise the input data
    X_train_mean = np.mean(X_train)
    X_train_std = np.std(X_train)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    # Convert the targets to column vectors
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # Define the MLP model
    input_size = X_train.shape[1]
    hidden_sizes = [10, 10]
    output_size = y_train.shape[1]
    mlp = MLP(input_size, hidden_sizes, output_size)

    # Training parameters
    num_epochs = 500
    learning_rate = 0.05

    # Track loss over epochs for plotting
    losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Calling forward pass
        outputs = mlp.forward(X_train.T)

        # Backprop and gradient descent step for paramater update
        gradients = mlp.backprop(X_train.T, y_train.T)
        mlp.update_parameters(gradients, learning_rate)

        # Compute and print Binary Cross-Entropy Loss
        eps = 1e-8
        loss = -np.mean(y_train.T * np.log(outputs + eps) + (1 - y_train.T) * np.log(1 - outputs + eps))
        losses.append(loss)

        if ((epoch + 1) % 100) == 0:
            print(f"Epoch {epoch + 1} - Loss: {loss}")
        
    # Testing
    test_outputs = mlp.forward(X_test.T)
    test_loss = -np.mean(y_test.T * np.log(test_outputs + eps) + (1 - y_test.T) * np.log(1 - test_outputs + eps))
    print(f"Test Loss: {test_loss}")

    # Plot the training loss curve
    plot_loss_curve(losses)

