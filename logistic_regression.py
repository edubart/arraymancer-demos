import numpy as np
import glob
from PIL import Image
import time

def timer(s):
  global start
  if len(s) > 0:
    now = time.time()
    print(s, (now - start) * 1000, "ms")
  start = time.time()

# Load dataset
def loadDataset(path):
  cats = [np.array(Image.open(fname)) for fname in glob.glob(path + '/cat/*')]
  noncats = [np.array(Image.open(fname)) for fname in glob.glob(path + '/noncat/*')]
  inputs = np.concatenate((cats, noncats), 0)
  labels = np.concatenate((np.ones((len(cats))), np.zeros((len(noncats)))), 0)
  return [inputs, labels]

print("Loading data...")
timer('')
train_dataset_x, train_dataset_y = loadDataset("data/catvsnoncat/train")
test_dataset_x, test_dataset_y = loadDataset("data/catvsnoncat/test")
timer("Loaded data")

# Preprocess input data, all pixels for each image are flattened into
# column vectors, thus different columns stores different examples
num_features = train_dataset_x.shape[1] * train_dataset_x.shape[2] * train_dataset_x.shape[3]
m_train = train_dataset_x.shape[0]
m_test = test_dataset_x.shape[0]
train_x = train_dataset_x.reshape((m_train, num_features)).transpose().astype(np.float32)
train_y = train_dataset_y.reshape((1, m_train))
test_x = test_dataset_x.reshape((m_test, num_features)).transpose().astype(np.float32)
test_y = test_dataset_y.reshape((1, m_test))

# Simple pixel normalization
train_x /= 255.0
test_x /= 255.0

timer("Preprocessed data")

# Sigmoid function
def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

# Gradient descent optimize
def train(w, b, X, Y, max_iterations, learning_rate):
  m = X.shape[1]
  for i in range(1,max_iterations+1):
    # Forward propagation
    A = sigmoid(np.dot(w.transpose(), X) + b)
    cost = - np.sum((Y * np.log(A)) + ((1.0 - Y) * np.log(1.0 - A))) / m

    # Backward propagation
    difAY = A - Y
    dw = np.dot(X, difAY.transpose())
    db = np.sum(difAY)

    # Gradient descent
    w -= (learning_rate / m) * dw
    b -= (learning_rate / m) * db

    if i == 1 or i % 100 == 0:
      print("Cost after iteration ", i, ": ", cost)

# Prediction
def predict(w, b, X):
  A = sigmoid(np.dot(w.transpose(), X) + b)
  return np.where(A <= 0.5, 0, 1)

# Initialize weights with zeros
w = np.zeros((num_features, 1))
b = 0.0

# Print some information
print("Training inputs shape: ", train_dataset_x.shape)
print("Training labels shape: ", train_dataset_y.shape)
print("Test inputs shape: ", test_dataset_x.shape)
print("Test labels shape: ", test_dataset_y.shape)
print("Total number of parameters: ", num_features + 1)

# Train on data
print("Training...")
timer('')
train(w, b, train_x, train_y,
      max_iterations=2000,
      learning_rate=5e-3)
timer("Train finished!")

# Show accuracy
def score(pred_y, y):
  return (1.0 - np.mean(np.abs(pred_y - y))) * 100.0

train_pred_y = predict(w, b, train_x)
test_pred_y = predict(w, b, test_x)
train_accuracy = score(train_pred_y, train_y)
test_accuracy = score(test_pred_y, test_y)

print("Train accuracy: ", train_accuracy, "%")
print("Test accuracy: ", test_accuracy, "%")