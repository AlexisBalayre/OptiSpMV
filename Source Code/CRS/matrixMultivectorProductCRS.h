#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRS_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRS_H

#include "../MatrixDefinitions.h"
#include <chrono>
#include <vector>
#include <iostream>
#include <numeric>  
#include <algorithm>

/**
 * @brief Perform the matrix-vector multiplication in the CRS format
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductCRS(const SparseMatrixCRS &sparseMatrix,
                                   const FatVector &fatVector, FatVector &result, const int testNumber);

#endif
