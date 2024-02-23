#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRS_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRS_H

#include "../MatrixDefinitions.h"
#include <chrono>
#include <vector>
#include <iostream>
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
double matrixMultivectorProductCRS(const SparseMatrixCRS &sparseMatrix,
                                   const FatVector &fatVector, FatVector &result, const int testNumber);

#endif
