#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYELLPACK_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYELLPACK_H

#include "../MatrixDefinitions.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <numeric>   // For std::accumulate
#include <algorithm> // For std::sort

double matrixMultivectorProductELLPACK(const SparseMatrixELLPACK &sparseMatrix,
                                       const FatVector &fatVector, FatVector &result, const int testNumber);

#endif