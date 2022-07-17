import numpy as np

class MultipleLinearRegression:

    def __init__(self, learning_rate, num_iterations, dev_x, dev_y) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.rows, self.columns = dev_x.shape
        self.dev_x = dev_x
        self.dev_y = dev_y

    def forward_propagation(self, weights, bias_term):
        y_prediction = self.dev_x.dot(weights) + bias_term
        return y_prediction

    def compute_cost(self,dev_y, pred_y):
        first_term = 1/(2*self.rows)
        second_term = np.sum((dev_y - pred_y) ** 2)
        return first_term*second_term
    
    def backward_propagation(self, pred_y, dev_x, dev_y, weights, bias_term):
        dW = -(2* (dev_x.T).dot(dev_y - pred_y)) / self.rows
        db = -(2* np.sum(dev_y - pred_y)) / self.rows

        updated_weights = weights - (self.learning_rate * dW)
        updated_bias_term = bias_term - (self.learning_rate * db)

        return updated_weights, updated_bias_term

    def predict(self, x, weights, bias_term):
        return x.dot(weights) + bias_term

    