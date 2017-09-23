# Arraymancer MKL
#nim c -d:release --hints:off \
#  --passC:"-march=native -Ofast" \
#  --define:blas=mkl_intel_lp64 \
#  --passL:"-lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -flto" \
#  -d:blis -r logistic_regression.nim

# Arraymancer OpenBLAS
echo "Running arraymancer OpenBLAS:"
nim c -d:release --hints:off --passC:"-march=native -Ofast" \
  --passL:"-flto" \
  -d:blis -r logistic_regression.nim

# Numpy
echo "Running numpy benchmark:"
python3 logistic_regression.py

# Torch
echo "Running numpy benchmark:"
th logistic_regression.lua
