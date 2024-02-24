#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYELLPACKCUDA_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYELLPACKCUDA_H

#include "../MatrixDefinitions.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort

/**
 * @brief Multiply sparse matrix with a fat vector using ELLPACK format on CUDA
 *
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Fat vector
 * @param result Result vector
 * @param testNumber Number of times to run the operation
 * @param xBlockSize X-dimension block size
 * @param yBlockSize Y-dimension block size
 * @return double Performance in GFLOPS
 */
double matrixMultivectorProductELLPACKCUDA(const SparseMatrixELLPACK &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize);

#endif
