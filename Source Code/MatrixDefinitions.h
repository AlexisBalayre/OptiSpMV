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
    std::vector<double> values;  // Valeurs non-nulles de la matrice
    std::vector<int> colIndices; // Indices de colonnes pour les valeurs correspondantes
    std::vector<int> rowPtr;     // Pointeurs de début de ligne dans `values` et `colIndices`
    int numRows;                 // Nombre de lignes dans la matrice
    int numCols;                 // Nombre de colonnes dans la matrice
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
    std::vector<double> values;  // Valeurs non-nulles de la matrice, stockées linéairement
    std::vector<int> colIndices; // Indices de colonnes, stockés linéairement
    int maxNonZerosPerRow;       // Maximum de valeurs non-nulles par ligne
    int numRows;                 // Nombre de lignes dans la matrice
    int numCols;                 // Nombre de colonnes dans la matrice
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
    std::vector<double> values; // Valeurs du vecteur
    int numRows;                // Nombre de lignes dans le vecteur
    int numCols;                // Nombre de colonnes dans le vecteur
};

#endif
