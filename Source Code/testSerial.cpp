#include "utils.h" // Utility functions
#include "MatrixDefinitions.h"
#include "CRS/matrixMultivectorProductCRS.h"
#include "ELLPACK/matrixMultivectorProductELLPACK.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream> // Include for std::ifstream

int main()
{
    // Load the matrix
    std::string filepath = "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/dc1.mtx";

    // Number of values in the vector
    int k = 3;

    // Number of iterations for the test
    int testNumber = 10;

    // Use ifstream to open the file
    try
    {
        SparseMatrixCRS sparseMatrixCRS;
        SparseMatrixELLPACK sparseMatrixELLPACK;
        FatVector fatVector, resultCRS, resultELLPACK;

        // Read the matrix from the file
        readMatrixMarketFile(filepath, sparseMatrixCRS);

        // Convert the CRS matrix to ELLPACK
        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK);

        // Generate a random vector
        generateFatVector(fatVector, sparseMatrixCRS.numCols, k);

        // Perform the matrix-vector multiplication (CRS)
        resultCRS.values.resize(sparseMatrixCRS.numRows * k, 0.0);
        resultCRS.numCols = k;
        resultCRS.numRows = sparseMatrixCRS.numRows;
        double performance_crs = matrixMultivectorProductCRS(sparseMatrixCRS, fatVector, resultCRS, testNumber);

        // Perform the matrix-vector multiplication (ELLPACK)
        resultELLPACK.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
        resultELLPACK.numCols = k;
        resultELLPACK.numRows = sparseMatrixELLPACK.numRows;
        double performance_ellpack = matrixMultivectorProductELLPACK(sparseMatrixELLPACK, fatVector, resultELLPACK, testNumber);

        // Compare the results
        if (areMatricesEqual(resultCRS, resultELLPACK, 1e-6))
        {
            std::cout << "The results are equal." << std::endl;
        }
        else
        {
            std::cout << "The results are not equal." << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}