import numpy as np
import cupy as cp
import util

class Activation():
    """
    The Activation function class.
    
    Includes forward and derivative terms.

    Activation: sigmoid, tanh, opt_sig, ReLU, leaky_ReLU, softmax.
    """
    
    def __init__(self, activation_type = "sigmoid", hardware_type='CPU'):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "opt_sig", "ReLU", "leaky_ReLU", "softmax"]:   
            # softmax layer is used for output network result
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # if the model is using within cpu or gpu
        self.hardware_type = hardware_type

    def __call__(self, z):
        """
        This method allows instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "opt_sig":
            return self.opt_sig(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "leaky_ReLU":
            return self.leaky_ReLU(z)

        elif self.activation_type == "softmax":
            return self.softmax(z)

    def derivative(self, z):
            """
            Compute the derivative of each activation function.
            """
            if self.activation_type == "sigmoid":
                return self.derivative_sigmoid(z)

            elif self.activation_type == "tanh":
                return self.derivative_tanh(z)

            elif self.activation_type == "opt_sig":
                return self.derivative_opt_sig(z)

            elif self.activation_type == "ReLU":
                return self.derivative_ReLU(z)

            elif self.activation_type == "leaky_ReLU":
                return self.derivative_leaky_ReLU(z)

            elif self.activation_type == "softmax":
                return self.derivative_softmax(z)

    def sigmoid(self, x):
        """
        Implement the sigmoid activation.
        """
        if self.hardware_type == 'CPU':
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + cp.exp(-x))

    def tanh(self, x):
        """
        Implement tanh.
        """
        if self.hardware_type == 'CPU':
            return np.tanh(x)
        else:
            return cp.tanh(x)
        
    def opt_sig(self, x):
        """
        Implement optimized sigmoid function.
        """
        if self.hardware_type == 'CPU':
            return 1.7159 * np.tanh(2/3 * x)
        else:
            return 1.7159 * cp.tanh(2/3 * x)

    def ReLU(self, x):
        """
        Implement ReLU.
        """
        if self.hardware_type == 'CPU':
            return np.maximum(0, x)
        else: 
            return cp.maximum(0, x)

    def leaky_ReLU(self, x):
        """
        Implement leaky_ReLU.
        """
        if self.hardware_type == 'CPU':
            return np.maximum(0.01*x, x)
        else:
            return cp.maximum(0.01*x, x)
    
    def softmax(self, x):
        """ 
        Implement softmax.
        Take care of the overflow condition.
        """
        if self.hardware_type == 'CPU':
            temp = np.exp(x - np.max(x))
            return np.divide(temp, temp.sum(axis=1).reshape(-1, 1))
        else:
            temp = cp.exp(x - cp.max(x))
            return cp.divide(temp, temp.sum(axis=1).reshape(-1, 1))

    def derivative_sigmoid(self, x):
        """
        Derivative function of sigmoid
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def derivative_tanh(self, x):
        """
        Derivative function of tanh
        """
        return 1 - self.tanh(x) ** 2

    def derivative_opt_sig(self, x):
        """
        Derivative function of optimized sigmoid
        """
        return 1.7159 * (2/3) * (1 - self.tanh(2/3 * x))
        
    def derivative_ReLU(self, x):
        """
        Derivative function of 
        """