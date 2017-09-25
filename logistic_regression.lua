require 'torch'
require 'image'

-- Utilities
local function mergeTables(t1, t2)
  local t ={}
  for k,v in pairs(t1) do table.insert(t, v) end
  for k,v in pairs(t2) do table.insert(t, v) end
  return t
end
local function tableToTensor(tensors)
  local t = torch.Tensor()
  for i=1,#tensors do
    local x = tensors[i]
    x = x:reshape(1, unpack(x:size():totable()))
    t = torch.cat(t, x, 1)
  end
  return t
end
local function loadFromDir(path)
  local t = {}
  for file in paths.files(path) do
    if file:match('.png') then
      table.insert(t, image.load(path .. '/' .. file))
    end
  end
  return t
end

-- Set default tensor type to float32
torch.setdefaulttensortype('torch.FloatTensor')

-- Benchmarking timer
local ptimer = torch.Timer()
local function timer(what)
  if what then
    local now = os.time()
    print(what, " [", (ptimer:time().real)* 1000, "ms]")
  end
  ptimer:reset()
end

-- Load dataset
local function loadDataset(path)
  local cats = loadFromDir(path .. "/cat")
  local noncats = loadFromDir(path .. "/noncat")
  local inputs = tableToTensor(mergeTables(cats, noncats))
  local labels = torch.cat(torch.ones(#cats), torch.zeros(#noncats))
  return inputs, labels
end

print "Loading data..."
timer()
local train_dataset_x, train_dataset_y = loadDataset("data/catvsnoncat/train")
local test_dataset_x, test_dataset_y = loadDataset("data/catvsnoncat/test")
timer("Loaded data")

-- Preprocess input data, all pixels for each image are flattened into columns
local num_features = train_dataset_x:size()[2] * train_dataset_x:size()[3] * train_dataset_x:size()[4]
local m_train = train_dataset_x:size()[1]
local m_test = test_dataset_x:size()[1]
local train_x = train_dataset_x:reshape(m_train, num_features):t()
local train_y = train_dataset_y:reshape(1, m_train)
local test_x = test_dataset_x:reshape(m_test, num_features):t()
local test_y = test_dataset_y:reshape(1, m_test)

-- Already normalized to 0 - 1.0 range
--train_x:div(255.0)
--test_x:div(255.0)
timer("Preprocessed data")

-- Sigmoid function
local function apply_sigmoid(x)
  return x:neg():exp():add(1.0):cinv()
end

-- Model traning function
local function train(w, b, X, Y, max_iterations, learning_rate)
  local m = X:size()[2]
  for i=1,max_iterations do
    -- Forward propagation
    local A = apply_sigmoid((w:t() * X):add(b))
    local cost = - torch.cmul(Y, torch.log(A)):add(torch.cmul(1.0 - Y, (1.0 - A):log())):sum() / m

    -- Backward propagation
    local difAY = A - Y
    local db = difAY:sum()
    local dw = X * difAY:t()

    -- Gradient descent
    w:csub((learning_rate / m) * dw)
    b = b - (learning_rate / m) * db

    if i == 1 or i % 100 == 0 then
      print("Cost after iteration ", i, ": ", cost)
    end
  end

  return w, b
end

-- Prediction
local function predict(w, b, X)
  local A = apply_sigmoid((w:t() * X):add(b))
  return torch.ge(A, 0.5):float()
end

-- Initialize weights with zeros
local w = torch.zeros(num_features, 1)
local b = 0.0

local useCuda = false
if useCuda then
  require 'cutorch'
  w = w:cuda()
  train_x = train_x:cuda()
  train_y = train_y:cuda()
  test_x = test_x:cuda()
  test_y = test_y:cuda()
  timer("Loaded tensors into CUDA")
end

-- Print some information
print("Training inputs shape: ", unpack(train_dataset_x:size():totable()))
print("Training labels shape: ", unpack(train_dataset_y:size():totable()))
print("Test inputs shape: ", unpack(test_dataset_x:size():totable()))
print("Test labels shape: ", unpack(test_dataset_y:size():totable()))
print("Total number of parameters: ", num_features + 1)

-- Train on data
print "Training..."
timer()
w, b = train(w, b, train_x, train_y, 2000, 5e-3)
timer("Train finished!")

-- Show accuracy
local function score(pred_y, y)
  return (1.0 - torch.mean(torch.abs(pred_y - y))) * 100.0
end

local train_pred_y = predict(w, b, train_x)
local test_pred_y = predict(w, b, test_x)
local train_accuracy = score(train_pred_y, train_y:float())
local test_accuracy = score(test_pred_y, test_y:float())

print("Train accuracy: ", train_accuracy, "%")
print("Test accuracy: ", test_accuracy, "%")
