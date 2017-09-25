import sequtils, times, future, math
import arraymancer, arraymancer_vision

# Benchmarking timer
proc timer*(what: string = "") =
  var start {.global.} = cpuTime()
  if what.len > 0:
    echo what, " [", (cpuTime() - start)* 1000, "ms]"
  start = cpuTime()

# Load dataset
proc loadDataset(path: string): auto =
  let cats = loadFromDir(path & "/cat")
  let noncats = loadFromDir(path & "/noncat")
  let inputs = concat(cats, noncats).stack()
  let labels = concat(ones([cats.len], float32), zeros([noncats.len], float32), 0)
  return (inputs, labels)

echo "Loading data..."
timer()
var (train_dataset_x, train_dataset_y) = loadDataset("data/catvsnoncat/train")
var (test_dataset_x, test_dataset_y) = loadDataset("data/catvsnoncat/test")
timer("Loaded data")

# Preprocess input data, all pixels for each image are flattened into columns
let num_features = train_dataset_x.shape[1] * train_dataset_x.shape[2] * train_dataset_x.shape[3]
let m_train = train_dataset_x.shape[0]
let m_test = test_dataset_x.shape[0]
var train_x = train_dataset_x.unsafeReshape([m_train, num_features]).unsafeTranspose().asContiguous().astype(float32)
var train_y = train_dataset_y.unsafeReshape([1, m_train]).asContiguous().astype(float32)
var test_x = test_dataset_x.unsafeReshape([m_test, num_features]).unsafeTranspose().asContiguous().astype(float32)
var test_y = test_dataset_y.unsafeReshape([1, m_test]).asContiguous().astype(float32)

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
  var params = newSeq[LayerParams]()
  for i in 1..<layerSizes.len:
    var W = randomTensor([layerSizes[i], layerSizes[i-1]], 2.0).astype(float32)
    W .-= 1.0f
    W /= sqrt(layerSizes[i-1].float32)
    let b = zeros([layerSizes[i], 1], float32)
    params.add((W, b))
  return params

# Sigmoid forward and backward
proc sigmoidForward(Z: Tensor[float32]): auto {.inline.} =
  let A = Z.map(x => 1.0f/(1.0f + exp(-x)))
  let A_prev = A
  return (A, A_prev)

proc sigmoidBackward(dA: Tensor[float32], A_prev: Tensor[float32]): auto {.inline.} =
  let dZ = dA .* A_prev.map(x => x * (1.0f - x))
  return dZ

# Relu forward and backward
proc reluForward(Z: Tensor[float32]): auto =
  let A = Z.map(x => max(0.0f, x))
  let Z_prev = Z
  return (A, Z_prev)

proc reluBackward(dA: Tensor[float32], Z_prev: Tensor[float32]): auto {.inline.} =
  proc relub(x: float32): float32 =
    if x <= 0.0f:
      return 0.0f
    return 1.0f
  let dZ = dA .* Z_prev.map(relub)
  return dZ

# Linear forward and backward
proc linearForward(A, W, b: Tensor[float32]): auto {.inline.} =
  let Z = (W * A) .+ b
  let A_prev = A
  return (Z, A_prev)

proc linearBackward(dZ, A_prev, W, b: Tensor[float32]): auto {.inline.} =
  let m = A_prev.shape[1].float32
  let dW = (1.0f/m) * (dZ * A_prev.unsafeTranspose())
  let db = (1.0f/m) * sum(dZ, 1)
  let dA_prev = W.unsafeTranspose() * dZ
  return (dA_prev, dW, db)

# Cost function forward and backward
proc crossEntropyForward(AL, Y: Tensor[float32]): float32 {.inline.} =
  let m = Y.shape[1].float32
  let cost = (-1.0f/m) * sum((Y .* ln(AL)) + ((1.0f .- Y) .* ln(1.0f .- AL)))
  return cost

proc crossEntropyBackward(AL, Y: Tensor[float32]): auto {.inline.} =
  let Y = Y.reshape(AL.shape)
  let dAL = - ((Y ./ AL) .- ((1.0f .- Y) ./ (1.0f .- AL)))
  return dAL

# Neural network forward and backward
proc networkForward(X: Tensor[float32], params: seq[LayerParams]): auto {.inline.} =
  var caches = newSeq[Tensor[float32]]()
  var Z, cache: Tensor[float32]
  var A = X
  for i in 0..<params.len:
    let W = params[i].W
    let b = params[i].b
    (Z, cache) = linearForward(A, W, b)
    caches.add(cache)
    if i == params.len-1: # Last layer activation
      (A, cache) = sigmoidForward(Z)
    else:
      (A, cache) = reluForward(Z)
    caches.add(cache)
  return (A, caches)

proc networkBackward(dAL: Tensor[float32], params: seq[LayerParams], caches: seq[Tensor[float32]]): auto {.inline.} =
  var grads = newSeq[LayerParams]()
  var dW, dB, dZ, cache: Tensor[float32]
  var dA = dAL
  var cacheIndex = caches.len-1
  for i in countdown(params.len-1,0):
    cache = caches[cacheIndex]; cacheIndex.dec
    if i == params.len-1: # Last layer activation
      dZ = sigmoidBackward(dA, cache)
    else:
      dZ = reluBackward(dA, cache)
    cache = caches[cacheIndex]; cacheIndex.dec
    (dA, dW, db) = linearBackward(dZ, cache, params[i].W, params[i].b)
    grads.insert((dW,dB))
  return grads

type
  AdamState = object
    learning_rate: float32
    beta1: float32
    beta2: float32
    epsilon: float32
    counter: int
    vgrads: seq[LayerParams]
    sgrads: seq[LayerParams]

# Adam optimizer
proc initializeAdam(layerSizes: openarray[int],
                    learning_rate = 0.001f,
                    beta1 = 0.9f,
                    beta2 = 0.999f,
                    epsilon = 1e-8f): AdamState =
  result.counter = 1
  result.learning_rate = learning_rate
  result.beta1 = beta1
  result.beta2 = beta2
  result.epsilon = epsilon
  result.vgrads = newSeq[LayerParams]()
  result.sgrads = newSeq[LayerParams]()
  for i in 1..<layerSizes.len:
    let vW = zeros([layerSizes[i], layerSizes[i-1]], float32)
    let vb = zeros([layerSizes[i], 1], float32)
    let sW = zeros([layerSizes[i], layerSizes[i-1]], float32)
    let sb = zeros([layerSizes[i], 1], float32)
    result.vgrads.add((vW, vb))
    result.sgrads.add((sW, sb))

proc optimizeAdam(params: var seq[LayerParams], grads: seq[LayerParams], state: var AdamState) {.inline.} =
  for i in 0..<params.len:
    # Moving average of the gradients
    state.vgrads[i].W = (state.vgrads[i].W * state.beta1) + (1.0f - state.beta1)*grads[i].W
    state.vgrads[i].b = (state.vgrads[i].b * state.beta1) + (1.0f - state.beta1)*grads[i].b

    # Compute bias correct first moment estimate
    let vW = state.vgrads[i].W / (1.0f - pow(state.beta1, state.counter.float32))
    let vb = state.vgrads[i].b / (1.0f - pow(state.beta1, state.counter.float32))

    # Moving average of the squared gradients
    state.sgrads[i].W = (state.sgrads[i].W * state.beta2) + (1.0f - state.beta2)*(grads[i].W .* grads[i].W)
    state.sgrads[i].b = (state.sgrads[i].b * state.beta2) + (1.0f - state.beta2)*(grads[i].b .* grads[i].b)

    # Compute bias corrected second moment estimate
    let sW = state.sgrads[i].W / (1.0f - pow(state.beta2, state.counter.float32))
    let sb = state.sgrads[i].b / (1.0f - pow(state.beta2, state.counter.float32))

    # Update paramaters
    params[i].W -= (state.learning_rate * (vW ./ (sqrt(sW) .+ state.epsilon)))
    params[i].b -= (state.learning_rate * (vb ./ (sqrt(sb) .+ state.epsilon)))

  state.counter = state.counter + 1

# Model traning function
proc trainModel(layerSizes: openarray[int], X, Y: Tensor[float32], max_iterations: int, optim: var AdamState): auto =
  var params = initializeParameters(layerSizes)
  for i in 1..max_iterations:
    # Forward propagation
    let (AL, caches) = networkForward(X, params)
    let cost = crossEntropyForward(AL, Y)
    echo cost

    # Backward propagation
    let dA = crossEntropyBackward(AL, Y)
    let grads = networkBackward(dA, params, caches)

    # Gradient descent
    optimizeAdam(params, grads, optim)

    if i == 1 or i mod 10 == 0:
      echo "Cost after iteration ", i, ": ", cost
  return params

# Prediction
proc predict(X: Tensor[float32], params: seq[LayerParams]): Tensor[float32] =
  let (AL, caches) = networkForward(X, params)
  return AL.map(x => (x >= 0.5f).float32)

# Print some information
echo "Training inputs shape: ", train_dataset_x.shape
echo "Training labels shape: ", train_dataset_y.shape
echo "Test inputs shape: ", test_dataset_x.shape
echo "Test labels shape: ", test_dataset_y.shape
echo "Total number of parameters: ", num_features + 1

# Train on data
echo "Training..."
timer()
let layers = [num_features,20,7,5,1]
var optimState = initializeAdam(layers)
let params = trainModel(
  [num_features,20,7,5,1],
  train_x, train_y,
  max_iterations=500,
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
