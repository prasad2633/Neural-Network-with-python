import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data
import math



nnfs.init()

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weigths = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weigths) + self.bias

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs , axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences = y_pred_clip[range(samples),y_true]
    
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clip*y_true, axis=1)

        negaive_log_likelihood = -np.log(correct_confidences)
        return negaive_log_likelihood
    


X, y = spiral_data(samples=100, classes=3)

dense1 = layer_dense(2, 3)
activation1 = Activation_ReLU()

dense2 = layer_dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss=", loss) 



