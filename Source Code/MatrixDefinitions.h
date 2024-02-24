#ifndef MATRIXDEFINITIONS_H
#define MATRIXDEFINITIONS_H

#include <vector>

/**
 * @brief Struct to represent a sparse matrix in Compressed Row Storage (CRS) format
 *
 * @param values  Non-zero values
 * @param colIndices  Column indices of non-zero values
 * @param rowPtr  Row pointers
 * @param numRows  Number of rows
 * @param numCols  Number of columns
 */
struct SparseMatrixCRS
{
    std::vector<double> values;
    std::vector<int> colIndices;
    std::vector<int> rowPtr;
    int numRows;
    int numCols;
};

/**
 * @brief Struct to represent a sparse matrix in ELLPACK format
 *
 * @param values  Non-zero values
 * @param colIndices  Column indices of non-zero values
 * @param maxNonZerosPerRow  Maximum number of non-zero values per row
 * @param numRows  Number of rows
 * @param numCols  Number of columns
 */
struct SparseMatrixELLPACK
{
    std::vector<double> values;
    std::vector<int> colIndices;
    int maxNonZerosPerRow;
    int numRows;
    int numCols;
};

/**
 * @brief Struct to represent a fat vector
 *
 * @param values  Matrix values
 * @param numRows  Number of rows
 * @param numCols  Number of columns
 */
struct FatVector
{
    std::vector<double> values;
    int numRows;
    int numCols;
};

#endif
