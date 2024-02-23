#ifndef MATRIXDEFINITIONS_H
#define MATRIXDEFINITIONS_H

#include <vector>

struct SparseMatrixCRS
{
    std::vector<double> values;  // Valeurs non-nulles de la matrice
    std::vector<int> colIndices; // Indices de colonnes pour les valeurs correspondantes
    std::vector<int> rowPtr;     // Pointeurs de début de ligne dans `values` et `colIndices`
    int numRows;                 // Nombre de lignes dans la matrice
    int numCols;                 // Nombre de colonnes dans la matrice
};

struct SparseMatrixELLPACK
{
    std::vector<double> values; // Valeurs non-nulles de la matrice, stockées linéairement
    std::vector<int> colIndices; // Indices de colonnes, stockés linéairement
    int maxNonZerosPerRow; // Maximum de valeurs non-nulles par ligne
    int numRows; // Nombre de lignes dans la matrice
    int numCols; // Nombre de colonnes dans la matrice
};


struct FatVector
{ 
    std::vector<double> values; // Valeurs du vecteur
    int numRows; // Nombre de lignes dans le vecteur
    int numCols; // Nombre de colonnes dans le vecteur
};

#endif
