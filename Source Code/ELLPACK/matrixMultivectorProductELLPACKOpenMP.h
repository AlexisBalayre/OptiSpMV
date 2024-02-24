#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYELLPACKOPENMP_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYELLPACKOPENMP_H

#include <iostream>
#include <vector>
#include <omp.h>     // Include OpenMP header
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort
#include "../MatrixDefinitions.h"

/**
 * @brief Perform the matrix-vector multiplication for an ELLPACK matrix using OpenMP
 *
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @param numThreads Number of threads to use
 * @param chunkSize Chunk size for the OpenMP parallel for
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductELLPACKOpenMP(const SparseMatrixELLPACK &sparseMatrix,
                                             const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize);

#endif
