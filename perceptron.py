import numpy as np
import matplotlib.pyplot as plt

# -------------------- Single Perceptron --------------------
class Perceptron():
    def __init__(self, inputs, function):
        self.i = inputs # number of input connections
        self.w = np.random.random(size = inputs)*2 - 1 # initialize weights randomly in [-1,1]
        self.b = np.random.random()*2 - 1 # initialize bias randomly in [-1,1]
        self.lr = np.random.random() # learning rate (randomized, could be tuned)
        self.a = function # activation function (e.g., sigmoid, step)

    # Compute the output of this perceptron for input X
    def forward(self, X):
        y = np.dot(self.w, X) + self.b # weighted sum + bias
        return self.a(y) # apply activation function

# -------------------- Multi-Layer Neural Network --------------------
class NeuralNet():
    def __init__(self, inputs, hidden, ouputs, function):
        self.i = inputs # number of input neurons
        self.h = hidden # list of hidden layer sizes
        self.o = ouputs # number of output neurons
        self.hls = []   # hidden layers (list of lists of perceptrons)
        
        # Initialize hidden layers
        for i in range(len(self.h)):
            hl = []
            for j in range(self.h[i]):
                if i == 0:
                    # First hidden layer connected to input
                    hl.append(Perceptron(self.i, function))
                else:
                    # Subsequent hidden layers connected to previous hidden layer
                    hl.append(Perceptron(self.h[i-1], function))
            self.hls.append(hl)
        
        self.hos = [] # store outputs of each hidden layer
        # Initialize output layer perceptrons connected to last hidden layer
        self.end = [Perceptron(self.h[-1], function) for i in range(self.o)]
        self.output = [0 for i in range(self.o)] # initialize outputs
        self.lr = 0.1 # fixed learning rate for backpropagation

    # -------------------- Extract Network Parameters --------------------
    def getParams(self):
        params = []
        # Hidden layers
        for hl in self.hls:
            for neuron in hl:
                w = np.array(neuron.w).flatten()
                params.extend([float(x) for x in w])
                params.append(float(neuron.b))
        # Output layer
        for neuron in self.end:
            w = np.array(neuron.w).flatten()
            params.extend([float(x) for x in w])
            params.append(float(neuron.b))
        return params

    # -------------------- Set Network Parameters from a Gene --------------------
    def setParams(self, gene):
        i = 0
        # Set hidden layer weights and biases
        for hl in self.hls:
            for neuron in hl:
                w_vals = gene[i: i + neuron.i] # extract weights
                i += neuron.i
                neuron.w = np.array(w_vals, dtype=float)
                neuron.b = float(gene[i]) # extract bias
                i += 1
        # Set output layer weights and biases
        for neuron in self.end:
            w_vals = gene[i: i + neuron.i]
            i += neuron.i
            neuron.w = np.array(w_vals, dtype=float)
            neuron.b = float(gene[i])
            i += 1

    # -------------------- Forward Pass --------------------
    def forward(self, X):
        ho = X
        self.hos = []  # reset hidden outputs
        # Compute outputs for each hidden layer
        for hl in self.hls:
            ho = [neuron.forward(ho) for neuron in hl]
            self.hos.append(ho)
        # Compute outputs of output layer
        for i in range(len(self.output)):
            self.output[i] = self.end[i].forward(self.hos[-1])
        return self.output
    
    # -------------------- Training via Backpropagation --------------------
    def train(self, X, target):
        self.forward(X) # forward pass
        
        # Compute derivative of sigmoid for output layer
        derivative = [self.output[i] * (1 - self.output[i]) for i in range(self.o)]
        # Compute output layer errors
        errorOutput = [derivative[i] * (target[i] - self.output[i]) for i in range(self.o)]
        errors = []
        error = errorOutput
        w = np.array([self.end[i].w for i in range(self.o)]) # weights from hidden to output

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.h))):
            d = np.array(self.hos[i])*(1 - np.array(self.hos[i])) # derivative of hidden outputs
            errors.append(d * np.dot(w.T, np.array(error).flatten())) # hidden layer error
            
            # Update hidden layer weights and biases
            if i == 0:
                for j in range(len(self.hls[i])):
                    self.hls[i][j].w += np.array(X)*errors[len(self.h) - 1 - i][j]*self.lr
                    self.hls[i][j].b += errors[len(self.h) - 1 - i][j]*self.lr
                w = np.zeros((len(self.hls[i]), self.i))
            else:
                for j in range(len(self.hls[i])):
                    self.hls[i][j].w += np.array(self.hos[i - 1])*errors[len(self.h) - 1 - i][j]*self.lr
                    self.hls[i][j].b += errors[len(self.h) - 1 - i][j]*self.lr
                w = np.zeros((len(self.hls[i]), len(self.hls[i - 1][0].w)))
            
            # Update weight matrix for next backprop step
            for j in range(len(self.hls[i])):
                for k in range(len(self.hls[i][j].w)):
                    w[j][k] = self.hls[i][j].w[k]
            error = errors[len(self.h) - 1 - i]

        # Update output layer weights and biases
        for i in range(self.o):
            self.end[i].w += np.array(self.hos[-1])*errorOutput[i]*self.lr
            self.end[i].b += errorOutput[i]*self.lr
        
        return np.mean(np.abs(errorOutput)) # return mean absolute error

    # -------------------- Visualization of Decision Boundary --------------------
    def visualize(self, dataset, target, density):
        X = np.linspace(-1.05, 1.05, density)
        Y = np.linspace(-1.05, 1.05, density)
        output = np.zeros((density, density))

        # Evaluate network over a grid
        i = 0
        for x in X:
            j = 0
            for y in Y:
                output[i][j] = np.mean(self.forward([x, y]))
                j += 1
            i+= 1
        
        # Plot decision boundary as contour
        plt.contourf(X, Y, output)
        plt.colorbar()
        plt.title(f"Learning rate: {self.lr}")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        # Overlay data points
        for i in range(len(dataset)):
            if target[i] == 1:
                plt.plot(dataset[i][0], dataset[i][1], 'wo') # class 1
            else:
                plt.plot(dataset[i][0], dataset[i][1], 'wx') # class 0
        plt.show()
