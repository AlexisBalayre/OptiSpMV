#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRSCUDA_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRSCUDA_H

#include "../MatrixDefinitions.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param fatVector Fat vector
 * @param vecCols  Number of columns in the dense vector
 * @return FatVector Result of the multiplication
 */
double matrixMultivectorProductCRSCUDA(const SparseMatrixCRS &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize);

#endif
