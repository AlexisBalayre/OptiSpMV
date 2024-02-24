#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYELLPACK_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYELLPACK_H

#include "../MatrixDefinitions.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort

/**
 * @brief Perform the matrix-vector multiplication for an ELLPACK matrix
 * 
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductELLPACK(const SparseMatrixELLPACK &sparseMatrix,
                                       const FatVector &fatVector, FatVector &result, const int testNumber);

#endif