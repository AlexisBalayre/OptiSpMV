#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCRSOPENMP_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCRSOPENMP_H

#include <iostream>
#include <vector>
#include <omp.h>     
#include <numeric>  
#include <algorithm> 
#include "../MatrixDefinitions.h"

/**
 * @brief Perform the matrix-vector multiplication in the CRS format using OpenMP
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector  Dense vector
 * @param result  Pointer to the result vector
 * @param testNumber  Number of iterations for the performance measurement
 * @param numThreads  Number of threads to use
 * @param chunkSize  Chunk size for the OpenMP parallel for
 * @return double  GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductCRSOpenMP(const SparseMatrixCRS &sparseMatrix,
                                         const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize);

#endif
