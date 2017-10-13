import sequtils, times, future, math, os
import arraymancer, arraymancer_vision

# Benchmarking timer
proc timer*(what: string = "") =
  var start {.global.} = epochTime()
  if what.len > 0:
    echo what, " [", (epochTime() - start)* 1000, "ms]"
  start = epochTime()

# Load dataset
proc loadDataset(path: string): auto =
  let cats = loadFromDir(path & "/cat")
  let noncats = loadFromDir(path & "/noncat")
  let inputs = concat(cats, noncats).stack()
  let labels = concat(ones[uint8]([cats.len]), zeros[uint8]([noncats.len]), 0)
  return (inputs, labels)

echo "Loading data..."
timer()
var (train_dataset_x, train_dataset_y) = loadDataset(currentSourcePath().parentDir().joinPath("data/catvsnoncat/train"))
var (test_dataset_x, test_dataset_y) = loadDataset(currentSourcePath().parentDir().joinPath("data/catvsnoncat/test"))
timer("Loaded data")

# Preprocess input data, all pixels for each image are flattened into columns
let num_features = train_dataset_x.shape[1] * train_dataset_x.shape[2] * train_dataset_x.shape[3]
let m_train = train_dataset_x.shape[0]
let m_test = test_dataset_x.shape[0]
var train_x = train_dataset_x.unsafeReshape([m_train, num_features]).unsafeTranspose().unsafeContiguous().astype(float32)
var train_y = train_dataset_y.unsafeReshape([1, m_train]).unsafeContiguous().astype(float32)
var test_x = test_dataset_x.unsafeReshape([m_test, num_features]).unsafeTranspose().unsafeContiguous().astype(float32)
var test_y = test_dataset_y.unsafeReshape([1, m_test]).unsafeContiguous().astype(float32)

# Simple pixel normalization
train_x /= 255.0f
test_x /= 255.0f
timer("Preprocessed data")

type
  LayerParams = tuple
    W: Tensor[float32]
    b: Tensor[float32]

# Parameters initialization
proc initializeParameters(layerSizes: openarray[int]): auto =
  var params = newSeqOfCap[LayerParams](layerSizes.len-2)
  for i in 1..<layerSizes.len:
    # Kaiming uniform initialization (He initialisation)
    let std = 1.0f/sqrt((layerSizes[i-1]).float32)
    let gain = sqrt(2.0f) # ReLU gain
    let bound = gain * std * sqrt(3.0f)
    var W = randomTensor([layerSizes[i], layerSizes[i-1]], -bound..bound)
    let b = zeros[float32]([layerSizes[i], 1])
    params.add((W, b))
  return params

# Sigmoid forward and backward
proc sigmoidForward(Z, cache: var Tensor[float32]) {.inline.} =
  Z.apply_inline():
    1.0f/(1.0f + exp(-x))
  cache = Z.unsafeView()

proc sigmoidBackward(dA: var Tensor[float32], cache: var Tensor[float32]) {.inline.} =
  dA.apply2_inline(cache):
    x * (y * (1.0f - y))

# Relu forward and backward
proc reluForward(Z, cache: var Tensor[float32]) =
  cache = Z
  Z.apply_inline(max(0.0f, x))

proc reluBackward(dA: var Tensor[float32], cache: var Tensor[float32]) {.inline.} =
  cache.apply_inline(if x <= 0.0f: 0.0f else: 1.0f)
  dA .*= cache

# Linear forward and backward
proc linearForward(A, cache: var Tensor[float32], W, b: Tensor[float32]) {.inline.} =
  cache = A.unsafeView()
  A = W * A
  A .+= b

proc linearBackward(dZ: var Tensor[float32], cache, W, b: Tensor[float32], dW, dB: var Tensor[float32], skip: bool) {.inline.} =
  let m = cache.shape[1].float32
  let factor = 1.0f/m
  let fdZ = factor * dZ
  dW = fdZ * cache.unsafeTranspose()
  db = sum(fdZ, 1)
  if not skip:
    dZ = W.unsafeTranspose() * dZ

# Cost function forward and backward
proc crossEntropyForward(A, Y: Tensor[float32]): float32 {.inline.} =
  let m = Y.shape[1].float32
  result = 0.0f
  for a, y in zip(A, Y):
    result += (y * ln(a)) + ((1.0f - y) * ln(1.0f - a))
  result *= -1.0f/m

proc crossEntropyBackward(dA: var Tensor[float32], A, Y: Tensor[float32]) {.inline.} =
  dA = map2_inline(A, Y):
    - (y / x) + ((1.0f - y) / (1.0f - x))

# Neural network forward and backward
proc networkForward(X: Tensor[float32], params: seq[LayerParams], A: var Tensor[float32], caches: var seq[Tensor[float32]]) {.inline.} =
  A = X.unsafeView()
  for i in 0..<params.len:
    linearForward(A, caches[i*2], params[i].W, params[i].b)
    if i == params.len-1: # Last layer activation
      sigmoidForward(A, caches[i*2+1])
    else:
      reluForward(A, caches[i*2+1])

proc networkBackward(dA: var Tensor[float32], params: seq[LayerParams], caches: var seq[Tensor[float32]], grads: var seq[LayerParams]) {.inline.} =
  for i in countdown(params.len-1,0):
    if i == params.len-1: # Last layer activation
      sigmoidBackward(dA, caches[2*i+1])
    else:
      reluBackward(dA, caches[2*i+1])
    linearBackward(dA, caches[2*i], params[i].W, params[i].b, grads[i].W, grads[i].b, i == 0)

type
  AdamState = object
    learning_rate: float32
    beta1: float32
    beta2: float32
    epsilon: float32
    weight_decay: float32
    counter: int
    m: seq[LayerParams]
    v: seq[LayerParams]

# Adam optimizer
proc initializeAdam(layerSizes: openarray[int],
                    learning_rate = 0.001f,
                    weight_decay = 0.0f,
                    beta1 = 0.9f,
                    beta2 = 0.999f,
                    epsilon = 1e-3f): AdamState =
  result.counter = 0
  result.learning_rate = learning_rate
  result.beta1 = beta1
  result.beta2 = beta2
  result.epsilon = epsilon
  result.weight_decay = weight_decay
  result.m = newSeqOfCap[LayerParams](layerSizes.len-2)
  result.v = newSeqOfCap[LayerParams](layerSizes.len-2)
  for i in 1..<layerSizes.len:
    let vW = zeros[float32]([layerSizes[i], layerSizes[i-1]])
    let vb = zeros[float32]([layerSizes[i], 1])
    let sW = zeros[float32]([layerSizes[i], layerSizes[i-1]])
    let sb = zeros[float32]([layerSizes[i], 1])

    result.m.add((vW, vb))
    result.v.add((sW, sb))


proc optimizeAdam(params: var seq[LayerParams], grads: var seq[LayerParams], state: var AdamState) {.inline.} =
  state.counter = state.counter + 1

  # Cache some variables
  let beta1 = state.beta1
  let beta2 = state.beta2
  let obeta1 = 1.0f - beta1
  let obeta2 = 1.0f - beta2
  let weight_decay = state.weight_decay
  let epsilon = state.epsilon
  let counter = state.counter.float32

  # Compute bias correcton
  let biasCorrection1 = 1.0f - pow(beta1, counter)
  let biasCorrection2 = 1.0f - pow(beta2, counter)

  # Calculate step size
  let step = state.learning_rate * (sqrt(biasCorrection2)/biasCorrection1)

  for i in 0..<params.len:
    # L2 Regularization
    var dW = grads[i].W.unsafeView()
    var db = grads[i].b.unsafeView()
    if weight_decay != 0.0f:
      dW = map2_inline(dW, params[i].W, x + (weight_decay * y))
      db = map2_inline(db, params[i].b, x + (weight_decay * y))

    # Moving average of the gradients
    proc adam_calcm(m, grad: var Tensor[float32], beta1, obeta1: float32) =
      m.apply2_inline(grad, (x * beta1) + (obeta1*y))
    adam_calcm(state.m[i].W, dW, beta1, obeta1)
    adam_calcm(state.m[i].b, db, beta1, obeta1)

    # Moving average of the squared gradients
    proc adam_calcv(v, grad: var Tensor[float32], beta2, obeta2: float32) =
      v.apply2_inline(grad, (x * beta2) + (obeta2*y*y))
    adam_calcv(state.v[i].W, dW, beta2, obeta2)
    adam_calcv(state.v[i].b, db, beta2, obeta2)

    # Update params
    proc adam_update_grads(param, m, v: var Tensor[float32], step, epsilon: float32) =
      param.apply3_inline(m, v, x - (step * (y / (sqrt(z) + epsilon))))
    adam_update_grads(params[i].W, state.m[i].W, state.v[i].W, step, epsilon)
    adam_update_grads(params[i].b, state.m[i].b, state.v[i].b, step, epsilon)


# Model traning function
proc trainModel(layerSizes: openarray[int], X, Y: Tensor[float32], max_iterations: int, optim: var AdamState): auto =
  var params = initializeParameters(layerSizes)
  var caches = newSeq[Tensor[float32]](2*params.len)
  var grads = newSeq[LayerParams](params.len)
  var A, dA: Tensor[float32]
  for i in 1..max_iterations:
    # Forward propagation
    networkForward(X, params, A, caches)
    let cost = crossEntropyForward(A, Y)

    # Backward propagation
    crossEntropyBackward(dA, A, Y)
    networkBackward(dA, params, caches, grads)

    # Gradient descent
    optimizeAdam(params, grads, optim)

    if i == 1 or i mod 10 == 0:
      echo "Cost after iteration ", i, ": ", cost
  return params

# Prediction
proc predict(X: Tensor[float32], params: seq[LayerParams]): Tensor[float32] {.noInit.} =
  var caches = newSeq[Tensor[float32]](2*params.len)
  var A: Tensor[float32]
  networkForward(X, params, A, caches)
  return A.map_inline((x >= 0.5f).float32)

# Print some information
echo "Training inputs shape: ", train_dataset_x.shape
echo "Training labels shape: ", train_dataset_y.shape
echo "Test inputs shape: ", test_dataset_x.shape
echo "Test labels shape: ", test_dataset_y.shape

# Train on data
echo "Training..."
timer()
let layers = [num_features,16,8,4,1]
var optimState = initializeAdam(layers,
  learning_rate=1e-3f,
  weight_decay=1e-3)
let params = trainModel(
  layers,
  train_x, train_y,
  max_iterations=200,
  optimState)
timer("Train finished!")

# Show accuracy
proc score(pred_y, y: Tensor[float32]): float32 =
  return (1.0f - mean(abs(pred_y - y))) * 100.0f

let train_pred_y = predict(train_x, params)
let test_pred_y = predict(test_x, params)
let train_accuracy = score(train_pred_y, train_y)
let test_accuracy = score(test_pred_y, test_y)

echo "Train accuracy: ", train_accuracy, "%"
echo "Test accuracy: ", test_accuracy, "%"

# Visualize dataset samples with visdom
try:
  let vis = newVisdomClient()
  vis.image(tile_collection(train_dataset_x.unsafeSlice(0..<16, _, _, _)), "Train Samples")
  vis.image(tile_collection(test_dataset_x.unsafeSlice(0..<16, _, _, _)), "Test Samples")
  for i in 0..<test_dataset_x.shape[0]:
    var text = if test_pred_y[0, i] == 1.0f: "is cat" else: "non cat"
    let img = test_dataset_x.unsafeAt(i, _, _, _)
    vis.image(img, "test" & $i, text)
  timer("Data visualization sent to visdom")
except:
  echo "Visualization not shown because visdom was not reacheable!"
