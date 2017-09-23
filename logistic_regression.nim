import sequtils, times, future, random
import arraymancer, arraymancer_vision

# Utilities
proc unsafeUnsqueeze*(t: Tensor, axis: int): Tensor =
  var shape = t.shape
  shape.insert(1, axis)
  t.reshape(shape)
proc toTensor*[T](s: openarray[Tensor[T]]): Tensor[T] =
  s.map(proc(t: Tensor[T]): Tensor[T] = t.unsafeUnsqueeze(0)).concat(0)
proc abs*[T](t: Tensor[T]): Tensor[T] =
  t.map(proc(x: T):T = abs(x))

# Benchmarking timer
proc timer*(what: string = "") =
  var start {.global.} = cpuTime()
  if what.len > 0:
    let now = cpuTime()
    echo what, " [", (now - start)* 1000, "ms]"
  start = cpuTime()

# Load dataset
proc loadDataset(path: string): auto =
  let
    cats = loadFromDir(path & "/cat")
    noncats = loadFromDir(path & "/noncat")
    inputs = concat(cats, noncats).toTensor().astype(float32)
    labels = concat(ones([cats.len], float32), zeros([noncats.len], float32), 0)
  return (inputs, labels)

echo "Loading data..."
timer()
var (train_dataset_x, train_dataset_y) = loadDataset("data/catvsnoncat/train")
var (test_dataset_x, test_dataset_y) = loadDataset("data/catvsnoncat/test")
timer("Loaded data")

# Visualize dataset samples with visdom
try:
  let vis = newVisdomClient()
  vis.image(tile_collection(train_dataset_x.unsafeSlice(0..<16, _, _, _).astype(uint8)), "Train Samples")
  vis.image(tile_collection(test_dataset_x.unsafeSlice(0..<16, _, _, _).astype(uint8)), "Test Samples")
  timer("Data visualization sent to visdom")
except:
  echo "Visualization not shown because visdom was not reacheable!"

# Preprocess input data, all pixels for each image are flattened into
# column vectors, thus different columns stores different examples
let
  num_features = train_dataset_x.channels * train_dataset_x.height * train_dataset_x.width
  m_train = train_dataset_x.shape[0]
  m_test = test_dataset_x.shape[0]
var
  train_x = train_dataset_x.unsafeReshape([m_train, num_features]).unsafeTranspose().asContiguous()
  train_y = train_dataset_y.unsafeReshape([1, m_train]).asContiguous()
  test_x = test_dataset_x.unsafeReshape([m_test, num_features]).unsafeTranspose().asContiguous()
  test_y = test_dataset_y.unsafeReshape([1, m_test]).asContiguous()

# Simple pixel normalization
train_x /= 255.0f
test_x /= 255.0f

timer("Preprocessed data")

# Sigmoid function
proc sigmoid(x: float32): float32 {.noSideEffect.} =
  1.0f/(1.0f + exp(-x))
makeUniversalLocal(sigmoid)

# Gradient descent optimize
proc train(w: var Tensor[float32], b: var float32, X, Y: Tensor[float32],
           max_iterations: int,
           learning_rate: float32) =
  let m = X.shape[1].float32

  for i in 1..max_iterations:
    const optimized_version = false
    when optimized_version:
      # Forward propagation
      let sb = b
      var A = (w.unsafeTranspose() * X).map(x => sigmoid(x + sb))
      proc cross_entropy(a, y: float32): float32 {.noSideEffect.} =
        (y * ln(a)) + ((1.0f - y) * ln(1.0f - a))
      let cost = - fold2(A, 0.0f, (r, a, y) => r + cross_entropy(a, y), Y) / m
    else:
      # Forward propagation
      let A = sigmoid((w.unsafeTranspose() * X) .+ b)
      let cost = - sum((Y .* ln(A)) + ((1.0f .- Y) .* ln(1.0f .- A))) / m

    # Backward propagation
    let
      difAY = A - Y
      dw = X * difAY.unsafeTranspose()
      db = sum(difAY)

    # Gradient descent
    w -= (learning_rate / m) * dw
    b -= (learning_rate / m) * db

    if i == 1 or i mod 100 == 0:
      echo "Cost after iteration ", i, ": ", cost

# Prediction
proc predict(w: Tensor[float32], b: float32, X: Tensor[float32]): Tensor[float32] =
  let A = sigmoid((w.unsafeTranspose() * X) .+ b)
  A.map(x => (x >= 0.5).float32)

# Initialize weights with zeros
var
  w = zeros([num_features, 1], float32)
  b = 0.0f

# Print some information
echo "Training inputs shape: ", train_dataset_x.shape
echo "Training labels shape: ", train_dataset_y.shape
echo "Test inputs shape: ", test_dataset_x.shape
echo "Test labels shape: ", test_dataset_y.shape
echo "Total number of parameters: ", num_features + 1

# Train on data
echo "Training..."
train(w, b, train_x, train_y,
      max_iterations=2000,
      learning_rate=5e-3f)
timer("Train finished!")

# Show accuracy
proc score(pred_y, y: Tensor[float32]): float32 =
  (1.0f - mean(abs(pred_y - y))) * 100.0f

let
  train_pred_y = predict(w, b, train_x)
  test_pred_y = predict(w, b, test_x)
  train_accuracy = score(train_pred_y, train_y)
  test_accuracy = score(test_pred_y, test_y)

echo "Train accuracy: ", train_accuracy, "%"
echo "Test accuracy: ", test_accuracy, "%"

# Show some test examples
try:
  let vis = newVisdomClient()
  for i in 0..<test_dataset_x.shape[0]:
    var text: string
    if test_pred_y[0, i] == 1.0f:
      text = "is cat"
    else:
      text = "non cat"
    let img = test_dataset_x
      .unsafeSlice(i, _, _, _)
      .unsafeReshape(test_dataset_x.channels, test_dataset_x.height, test_dataset_x.width)
      .astype(uint8)
    vis.image(img, "test" & $i, text)
except:
  discard