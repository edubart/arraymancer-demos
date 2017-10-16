export _NUM_THREADS=4
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
export _DYNAMIC=FALSE

compile_nim='nim c --hints:off -d:release -d:openmp -d:mkl'
$compile_nim logistic_regression.nim
$compile_nim dnn_classification.nim

echo "[logistic regression] Running arraymancer"
for i in {1..10}; do
  ./logistic_regression | grep finished
done

echo "[logistic regression] Running numpy benchmark"
for i in {1..10}; do
  python3 logistic_regression.py | grep finished
done

echo "[logistic regression] Running torch benchmark"
for i in {1..10}; do
  th logistic_regression.lua | grep finished
done

echo "[dnn_classification] Running arraymancer"
for i in {1..10}; do
  ./dnn_classification | grep finished
done

echo "[dnn_classification] Running pytorch benchmark"
for i in {1..10}; do
  python3 dnn_classification.py | grep finished
done
