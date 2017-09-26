export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE

run_nim_mkl='nim c -d:release --hints:off --passC:"-march=native -Ofast -fopenmp" --define:blas=mkl_intel_lp64  --passL:"-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -flto -fopenmp" -r'
run_nim='nim c -d:release --hints:off --passC:"-march=native -Ofast -fopenmp" --passL:"-lpthread -flto -fopenmp" -r'

echo "[logistic regression] Running arraymancer (OpenBLAS)"
$run_nim logistic_regression.nim > /dev/null
$run_nim logistic_regression.nim
sleep 2

echo "[logistic regression] Running arraymancer (MKL)"
$run_nim_mkl logistic_regression.nim > /dev/null
$run_nim_mkl logistic_regression.nim
sleep 2

echo "[logistic regression] Running numpy benchmark"
python3 logistic_regression.py > /dev/null
python3 logistic_regression.py
sleep 2

echo "[logistic regression] Running torch benchmark"
th logistic_regression.lua > /dev/null
th logistic_regression.lua
sleep 2

echo "[dnn_classification] Running arraymancer (OpenBLAS)"
$run_nim dnn_classification.nim > /dev/null
$run_nim dnn_classification.nim
sleep 2

echo "[dnn_classification] Running arraymancer (MKL)"
$run_nim_mkl dnn_classification.nim > /dev/null
$run_nim_mkl dnn_classification.nim
sleep 2

echo "[dnn_classification] Running pytorch benchmark"
python3 dnn_classification.py > /dev/null
python3 dnn_classification.py
sleep 2
