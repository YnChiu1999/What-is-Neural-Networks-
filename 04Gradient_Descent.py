import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

learnrate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
# TODO: Calculate output# of neural network
nn_output = sigmoid(np.dot(x, w))

# error term (lowercase delta)
#error_term = error * sigmoid_prime(np.dot(x,weights))
# Gradient descent step 
#del_w = [ learnrate * error_term * x[0],
#                 learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x
# TODO: Calculate error of neural network
error = y-nn_output

# TODO: Calculate change in weights
del_w = learnrate * error * nn_output * (1 - nn_output) * x

print('Neural Network output:',nn_output)
#print(nn_output)
print('Amount of Error:',error)
print('Change in Weights:',del_w)