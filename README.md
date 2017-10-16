# Arraymancer Demos

This repo contains some arraymancer demos and benchmarks.

## Latest Benchmarks

These benchmarks share same model implementation and hyperparameters across
frameworks.

### Logistic regression

This benchmark consists of a classification on 209 RGB 3x64x64 images,
classifying them as cats or noncats. The model is a logistic unit.
Simple batch gradient descent is used.

#### CPU
| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer | OpenMP + MKL | **0.458ms**  |
| Torch7 | MKL | 0.686ms  |
| Numpy | MKL | 0.723ms  |

#### GPU
| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer | CUDA | WIP  |
| Torch7 | CUDA | 0.286ms |

### Deep neural network classification

As the above benchmark, consists of a classification on 209 RGB 3x64x64 images,
classifying them as cats or noncats. The model is a deep fully connected
neural network of layer sizes [209, 16, 8, 4, 1] (3 hidden layers + inputs/outputs layers).
The activation function for the hidden layers is ReLU, the layer layer activation function is Sigmoid,
and the loss is the binary cross entropy. Adam optimizer is used for batch gradient descent.

#### CPU
| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer | OpenMP + MKL | **2.907ms**  |
| PyTorch | MKL | 6.797ms  |

#### GPU
| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer | CUDA | WIP |
| PyTorch | CUDA | 4.765ms |

### Benchmark machine specs

* Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
* GeForce GTX 1080 Ti
* ArchLinux (kernel 4.12.13-1-ARCH, glibc 2.26)
* GCC 7.2.0
* MKL 2017.17.0.4.4
* OpenBLAS 0.2.20
* CUDA 8.0.61
* Nim 0.18.0 (head)
