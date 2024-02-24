#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRSCUDA_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRSCUDA_H

#include "../MatrixDefinitions.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>   
#include <algorithm> 

/**
 * @brief Perform matrix-vector product using CUDA
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector Input vector
 * @param result Output vector
 * @param testNumber Number of invocations
 * @param xBlockSize X-dimension of the block
 * @param yBlockSize Y-dimension of the block
 * @return double Performance in GFLOPS
 */
double matrixMultivectorProductCRSCUDA(const SparseMatrixCRS &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize);

#endif
