import sequtils, times, future
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

# Sigmoid function
proc sigmoid(x: float32): float32 {.noSideEffect.} =
  return 1.0f/(1.0f + exp(-x))

# Model traning function
proc train(w: var Tensor[float32], b: var float32, X, Y: Tensor[float32], max_iterations: int, learning_rate: float32) =
  let m = X.shape[1].float32
  for i in 1..max_iterations:
    # Forward propagation
    let A = ((w.unsafeTranspose() * X) .+ b).map(sigmoid)
    let cost = - sum((Y .* ln(A)) + ((1.0f .- Y) .* ln(1.0f .- A))) / m

    # Backward propagation
    let difAY = A - Y
    let dw = X * difAY.unsafeTranspose()
    let db = sum(difAY)

    # Gradient descent
    w -= (learning_rate / m) * dw
    b -= (learning_rate / m) * db

    if i == 1 or i mod 100 == 0:
      echo "Cost after iteration ", i, ": ", cost

# Prediction
proc predict(w: Tensor[float32], b: float32, X: Tensor[float32]): Tensor[float32] =
  let A = ((w.unsafeTranspose() * X) .+ b).map(sigmoid)
  return A.map(x => (x >= 0.5f).float32)

# Initialize weights with zeros
var w = zeros([num_features, 1], float32)
var b = 0.0f

# Print some information
echo "Training inputs shape: ", train_dataset_x.shape
echo "Training labels shape: ", train_dataset_y.shape
echo "Test inputs shape: ", test_dataset_x.shape
echo "Test labels shape: ", test_dataset_y.shape
echo "Total number of parameters: ", num_features + 1

# Train on data
echo "Training..."
timer()
train(w, b, train_x, train_y, max_iterations=2000, learning_rate=5e-3f)
timer("Train finished!")

# Show accuracy
proc score(pred_y, y: Tensor[float32]): float32 =
  return (1.0f - mean(abs(pred_y - y))) * 100.0f

let train_pred_y = predict(w, b, train_x)
let test_pred_y = predict(w, b, test_x)
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
