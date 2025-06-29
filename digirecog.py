import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the MNIST training data from CSV file
data = pd.read_csv('./data/train.csv')

# Convert the DataFrame to a NumPy array for faster computation
data = np.array(data)

# Show the shape of the data (number of samples, number of features+label)
data.shape

# Get the number of samples (m) and features (n)
m, n = data.shape

# Shuffle the data randomly (important for training)
np.random.shuffle(data)

# Split the data into development (dev) and training sets
# The first 1000 samples are for validation (dev), the rest for training
data_dev = data[0:1000].T
Y_dev = data_dev[0]         # Labels for dev set
X_dev = data_dev[1:n]       # Features for dev set

data_train = data[1000:m].T
Y_train = data_train[0]     # Labels for training set
X_train = data_train[1:n]   # Features for training set

# Normalize the pixel values to be between 0 and 1 (helps neural network training)
X_train = X_train / 255.
X_dev = X_dev / 255.

def init_params():
    """
    Initialize the weights and biases for a simple 2-layer neural network.
    - W1: weights for the first (hidden) layer, shape (10, 784)
    - b1: biases for the first layer, shape (10, 1)
    - W2: weights for the second (output) layer, shape (10, 10)
    - b2: biases for the second layer, shape (10, 1)
    We use small random values for weights and zeros for biases.
    """
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    """
    ReLU activation function: f(z) = max(0, z)
    This introduces non-linearity to the network.
    """
    return np.maximum(0, Z)

def sm(Z):
    """
    Softmax activation function for the output layer.
    Converts raw scores (logits) into probabilities that sum to 1.
    Formula: softmax(z_i) = exp(z_i) / sum_j exp(z_j)
    """
    expz = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for numerical stability
    return expz / np.sum(expz, axis=0, keepdims=True)

def fp(W1, b1, W2, b2, X):
    """
    Forward propagation: computes the output of the network.
    Steps:
    1. Z1 = W1 * X + b1         (linear for hidden layer)
    2. A1 = ReLU(Z1)            (activation for hidden layer)
    3. Z2 = W2 * A1 + b2        (linear for output layer)
    4. A2 = softmax(Z2)         (activation for output layer)
    Returns all intermediate values for use in backpropagation.
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sm(Z2)
    return Z1, A1, Z2, A2

def oh(Y):
    """
    One-hot encoding: converts labels (0-9) into vectors.
    For example, label 3 becomes [0,0,0,1,0,0,0,0,0,0].
    This is needed for the softmax output and loss calculation.
    """
    ohy = np.zeros((10, Y.size))
    ohy[Y, np.arange(Y.size)] = 1
    return ohy

def dReLU(Z):
    """
    Derivative of ReLU activation.
    Used in backpropagation to compute gradients.
    Returns 1 where Z > 0, else 0.
    """
    return (Z > 0).astype(float)

def bp(Z1, A1, Z2, A2, W2, X, Y):
    """
    Backpropagation: computes gradients for all weights and biases.
    Uses the chain rule to propagate the error backward.
    Steps:
    - dZ2: error at output layer (A2 - one-hot labels)
    - dW2, db2: gradients for output layer weights and biases
    - dZ1: error at hidden layer (using derivative of ReLU)
    - dW1, db1: gradients for hidden layer weights and biases
    Returns all gradients for parameter updates.
    """
    m = Y.size
    ohy = oh(Y)
    dZ2 = A2 - ohy
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def updtp(W1, b1, W2, b2, dW1, db1, dW2, db2, a):
    """
    Updates the weights and biases using gradient descent.
    Formula: parameter = parameter - learning_rate * gradient
    """
    W1 = W1 - a * dW1
    b1 = b1 - a * db1
    W2 = W2 - a * dW2
    b2 = b2 - a * db2
    return W1, b1, W2, b2

def gp(A2):
    """
    Get predictions from output probabilities.
    Returns the index (digit) with the highest probability for each sample.
    """
    return np.argmax(A2, 0)

def ga(predict, Y):
    """
    Calculates accuracy: the fraction of correct predictions.
    Formula: accuracy = (number of correct predictions) / (total samples)
    """
    return np.sum(predict == Y) / Y.size

def compute_loss(A2, Y):
    """
    Computes the cross-entropy loss for the predictions.
    Formula: L = -sum(y_true * log(y_pred)) / m
    Where y_true is one-hot encoded labels, y_pred is softmax output.
    """
    m = Y.size
    ohy = oh(Y)
    log_probs = np.log(A2 + 1e-8)  # add small value to avoid log(0)
    loss = -np.sum(ohy * log_probs) / m
    return loss

def gradient_descent(X, Y, iterations, a):
    """
    Trains the neural network using gradient descent.
    - X: input features
    - Y: labels
    - iterations: number of training steps
    - a: learning rate
    Prints accuracy and loss every 10 iterations.
    Returns the trained weights and biases.
    """
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = fp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = bp(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updtp(W1, b1, W2, b2, dW1, db1, dW2, db2, a)
        if i % 10 == 0:
            print("Iteration:", i)
            print("Accuracy:", ga(gp(A2), Y))
            print("Loss:", compute_loss(A2, Y))
    return W1, b1, W2, b2

def make_predicts(X, W1, b1, W2, b2):
    """
    Makes predictions for a set of inputs X using the trained network.
    Returns the predicted digit for each sample.
    """
    _, _, _, A2 = fp(W1, b1, W2, b2, X)
    predicts = gp(A2)
    return predicts

def test_prediction(index, W1, b1, W2, b2):
    """
    Tests the network on a single training sample.
    - Shows the image
    - Prints the predicted digit and the true label
    """
    cimg = X_train[:, index, None]
    prediction = make_predicts(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    cimg = cimg.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(cimg, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    # Train the neural network on the training data
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 1000, 0.1)

    # Evaluate on dev set
    dev_predictions = make_predicts(X_dev, W1, b1, W2, b2)
    print("Dev accuracy:", ga(dev_predictions, Y_dev))

    # Show predictions for a few samples
    for i in range(4):
        test_prediction(i, W1, b1, W2, b2)