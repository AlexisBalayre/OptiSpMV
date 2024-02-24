#ifndef UTILS_H
#define UTILS_H

#include <iostream>  // std::cout
#include <vector>    // std::vector
#include <cstdlib>   // rand() and srand()
#include <ctime>     // time()
#include <chrono>    // std::chrono::high_resolution_clock
#include <fstream>   // std::ifstream
#include <string>    // std::string
#include <sstream>   // std::stringstream
#include <utility>   // std::pair
#include <algorithm> // std::sort
#include <stdexcept> // std::runtime_error
#include <cmath>     // std::fabs
#include "MatrixDefinitions.h"

/**
 * Method to compare two matrices
 * @param mat1 First matrix
 * @param mat2 Second matrix
 * @param tolerance Tolerance for comparison
 * @return bool True if the matrices are equal, false otherwise
 */
bool areMatricesEqual(const FatVector &mat1, const FatVector &mat2, double tolerance);

/**
 * @brief Read a matrix from a Matrix Market file
 *
 * @param filename Filepath to the Matrix Market file
 * @param matrix Sparse matrix in CRS format (output)
 */
void readMatrixMarketFile(const std::string &filename, SparseMatrixCRS &matrix);

/**
 * @brief Generate a fat vector
 *
 * @param fatVector Pointer to the fat vector
 * @param numRows Number of rows in the fat vector
 * @param numCols Number of columns in the fat vector
 */
void generateFatVector(FatVector &fatVector, int numRows, int numCols);

/**
 * @brief Perform the matrix-vector multiplication for a CRS matrix
 *
 * @param crsMatrix Sparse matrix in CRS format
 * @param ellpackMatrix Pointer to the ELLPACK matrix (output)
 */
void convertCRStoELLPACK(const SparseMatrixCRS &crsMatrix, SparseMatrixELLPACK &ellpackMatrix);

#endif
