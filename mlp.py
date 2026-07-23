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
            self.biases.append(np.zeros(sizes[i],1))                                             # initialise to 0, per He convention
    
    # forward pass
    def forward(self, X):
        # Forward pass through the network
        self.activations = [X]
        self.z = [] # pre-activation value
        
        for i in range(self.num_layers):
            z = np.dot(self.weights[i], self.acativation[i]) + self.biases[i]
            self.z.append(z)
            if i < self.num_layers - 1:
                self.ReLU(z)            # ReLU activation function for hidden layers
            else:
                a = z                   # linear activations for output layer
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
        
        return gradients[::-1]  # reverse the gradients
    
    # Gradient descent step
    def update_parameters(self, gradients, learning_rate):
        for i in range(self.num_layers):
            self.weights -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][0]


    def ReLU(self, Z):
        return np.maximum(0, Z)
    
    def ReLUgradient(self, Z):
        return (Z > 0).astype(float)    # 1 where z > 0, else 0
        

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
