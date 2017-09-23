# Arraymancer Demos

This repo contains some arraymancer demos and benchmarks

### Latest Benchmarks results for logistic regression

Torch       CUDA      582ms
Torch       MKL       1417ms
Torch       OpenBLAS  13044ms
Numpy       MKL       17906ms
Arraymancer MKL       2325ms
Arraymancer OpenBLAS  12502ms

Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz
GeForce GTX 1080 Ti
ArchLinux (kernel 4.9.51-1-lts, glibc 2.26)
GCC 7.2.0
MKL 2017.17.0.4.4
OpenBLAS 0.2.20
CUDA 8.0.61