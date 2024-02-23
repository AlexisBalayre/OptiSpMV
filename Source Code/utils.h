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
 * Method to read a matrix from a Matrix Market file
 * @param filename Name of the file
 * @return SparseMatrixCRS Sparse matrix
 */
void readMatrixMarketFile(const std::string &filename, SparseMatrixCRS &matrix);

/**
 * Method to generate a random fat vector
 * @param n Number of rows
 * @param m Number of columns
 * @return FatVector Fat vector
 */
void generateFatVector(FatVector &fatVector, int numRows, int numCols);

void convertCRStoELLPACK(const SparseMatrixCRS &crsMatrix, SparseMatrixELLPACK &ellpackMatrix);



#endif
