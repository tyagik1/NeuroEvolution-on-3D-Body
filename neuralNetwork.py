import numpy as np

# -------------------- Activation Functions --------------------
# Sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Inverse of the sigmoid function
def inv_sigmoid(x):
    return np.log(x / (1-x))

# -------------------- Neural Network Class --------------------
class NeuralNetwork():

    def __init__(self, size, inputsize, outputsize):
        self.Size = size # total number of neurons in the network
        self.InputSize = inputsize # number of input neurons
        self.OutputSize = outputsize # number of output neurons
        self.Voltage = np.zeros(size) # neuron membrane potentials
        self.TimeConstants = np.ones(size) # time constant for each neuron
        self.Biases = np.zeros(size) # bias for each neuron
        self.Weights = np.zeros((size,size)) # recurrent weights between neurons
        self.SensorWeights = np.zeros((inputsize,size)) # weights from input sensors to neurons
        self.MotorWeights = np.zeros((size,outputsize)) # weights from neurons to output motors
        self.Output = np.zeros(size) # neuron outputs after activation
        self.Input = np.zeros(size) # input contribution to each neuron

    # -------------------- Random Initialization --------------------
    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size)) # recurrent weights
        self.SensorWeights = np.random.uniform(-10,10,size=(self.Size)) # input weights
        self.MotorWeights = np.random.uniform(-10,10,size=(self.Size)) # motor weights
        self.Biases = np.random.uniform(-10,10,size=(self.Size)) # biases
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.Size)) # time constants
        self.invTimeConstants = 1.0/self.TimeConstants # precompute inverse for efficiency

    # -------------------- Set Parameters from Genotype --------------------
    def setParameters(self,genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax):
        k = 0
        for i in range(self.Size):
            for j in range(self.Size):                          # Recurrent weight assignment
                self.Weights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.InputSize):
            for j in range(self.Size):                          # Sensor-to-neuron weights
                self.SensorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):
            for j in range(self.OutputSize):                    # Neuron-to-motor weights
                self.MotorWeights[i][j] = genotype[k]*WeightRange
                k += 1
        for i in range(self.Size):                              # Biases
            self.Biases[i] = genotype[k]*BiasRange
            k += 1
        for i in range(self.Size):                              # Time constants scaled to range
            self.TimeConstants[i] = ((genotype[k] + 1)/2)*(TimeConstMax-TimeConstMin) + TimeConstMin
            k += 1
        self.invTimeConstants = 1.0/self.TimeConstants # precompute inverse

    # -------------------- Initialize Network State --------------------
    def initializeState(self,v):
        self.Voltage = v # set neuron voltages
        self.Output = sigmoid(self.Voltage+self.Biases) # compute initial outputs

    # Initialize network from output values
    def initializeOutput(self,o):
        self.Output = o
        self.Voltage = inv_sigmoid(o) - self.Biases # compute corresponding voltages

    # -------------------- Network Step --------------------
    def step(self,dt,i):
        self.Input = np.dot(self.SensorWeights.T, i) # sensor input contribution
        netinput = self.Input + np.dot(self.Weights.T, self.Output) # total input from neurons
        # update voltages according to continuous-time dynamics
        self.Voltage += dt * (self.invTimeConstants*(-self.Voltage+netinput))
        self.Output = sigmoid(self.Voltage+self.Biases) # compute neuron outputs

    # Compute motor outputs
    def out(self):
        return sigmoid(np.dot(self.MotorWeights.T, self.Output))

    # -------------------- Save/Load --------------------
    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, sensorweights=self.SensorWeights, motorweights=self.MotorWeights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.SensorWeights = params['sensorweights'] 
        self.MotorWeights = params['motorweights'] 
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants
