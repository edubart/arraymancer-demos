import numpy as np
import glob
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def timer(s):
    global start
    if len(s) > 0:
        print(s, (time.time() - start) * 1000, "ms")
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

# Preprocess input data, all pixels for each image are flattened into columns
num_features = train_dataset_x.shape[1] * train_dataset_x.shape[2] * train_dataset_x.shape[3]
m_train = train_dataset_x.shape[0]
m_test = test_dataset_x.shape[0]
train_x = torch.from_numpy(train_dataset_x.reshape((m_train, num_features)).astype(np.float32))
train_y = torch.from_numpy(train_dataset_y.reshape((m_train, 1)).astype(np.float32))
test_x = torch.from_numpy(test_dataset_x.reshape((m_test, num_features)).astype(np.float32))
test_y = torch.from_numpy(test_dataset_y.reshape((m_test, 1)).astype(np.float32))

# Simple pixel normalization
train_x /= 255.0
test_x /= 255.0
timer("Preprocessed data")

# Model definition
class NetworkModel(nn.Module):
    def __init__(self, layerSizes):
        super(NetworkModel, self).__init__()
        layers = []
        for i in range(1, len(layerSizes)):
            layers.append(nn.Linear(layerSizes[i-1], layerSizes[i]))
            if i == len(layerSizes)-1: # last layer
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Model traning function
def train(model, criterion, optim, X, Y, max_iterations):
    for i in range(1,max_iterations+1):
        inputs = Variable(X, requires_grad=False)
        targets = Variable(Y, requires_grad=False)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        cost = loss.data[0]
        loss.backward()
        optimizer.step()

        if i == 1 or i % 2 == 0:
            print("Cost after iteration ", i, ": ", cost)

# Print some information
print("Training inputs shape: ", train_dataset_x.shape)
print("Training labels shape: ", train_dataset_y.shape)
print("Test inputs shape: ", test_dataset_x.shape)
print("Test labels shape: ", test_dataset_y.shape)

# Initialize model and loss
model = NetworkModel([num_features,16,8,4,1])
model.train()
criterion = nn.BCELoss(size_average=True)
optimizer = optim.Adam(model.parameters(),
    lr=1e-3,
    weight_decay=1e-3)
model.train()

# Train on data
print("Training...")
timer('')
train(model, criterion, optim, train_x, train_y, max_iterations=200)
timer("Train finished!")

# Prediction
def predict(model, X):
    model.eval()
    A = model(Variable(X, requires_grad=False)).data.numpy()
    return np.where(A <= 0.5, 0, 1)

# Show accuracy
def score(pred_y, y):
    return (1.0 - np.mean(np.abs(pred_y - y.numpy()))) * 100.0

train_pred_y = predict(model, train_x)
test_pred_y = predict(model, test_x)
train_accuracy = score(train_pred_y, train_y)
test_accuracy = score(test_pred_y, test_y)

print("Train accuracy: ", train_accuracy, "%")
print("Test accuracy: ", test_accuracy, "%")
