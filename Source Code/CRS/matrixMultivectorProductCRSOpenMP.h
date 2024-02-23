#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRSOPENMP_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRSOPENMP_H

#include <iostream>
#include <vector>
#include <omp.h>     // Include OpenMP header
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort
#include "../MatrixDefinitions.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param fatVector Fat vector
 * @param vecCols  Number of columns in the dense vector
 * @return FatVector Result of the multiplication
 */
double matrixMultivectorProductCRSOpenMP(const SparseMatrixCRS &sparseMatrix,
                                         const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize);

#endif
