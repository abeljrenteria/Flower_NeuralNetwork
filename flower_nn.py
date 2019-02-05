import numpy as np

# x - [length, width]
# y - [blue/red]
X = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1]), dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float)
xPred = np.array(([0.25, 1]), dtype=float)

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid funcction
def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.y = y
        self.weights = np.random.rand(self.input.shape[1], 1)
        self.bias = np.full((self.input.shape[0], 1), np.random.randn())
        self.output = np.zeros(y.shape)
  
    def feedforward(self):
        self.layer1 = sigmoid((np.dot(self.input, self.weights)) + self.bias)
        return self.layer1
       
    def backprop(self):
        # application of the chain rule to find the derivative of the loss function w.r.t weights1 and weights2
        d_weights = np.dot(self.input.T, 2*(self.y -self.layer1)*sigmoid_derivative(self.layer1))
        d_bias = np.dot(2*(self.y -self.layer1)*sigmoid_derivative(self.layer1), 1)

        #update weights with the slope of the loss function
        self.weights += d_weights
        self.bias += d_bias
  
    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
    
    # predictis if xPred input is a red or blue flower    
    def predict(self):
        pred = sigmoid(np.dot(xPred, self.weights))
        print (pred)
        if pred < 0.5:
            print ("The flower is Blue")
        else:
            print ("The flower is Red")

NN = NeuralNetwork(X, y)

# Trains the nn
for i in range (50000):   
    NN.train(X,y)

NN.predict()





