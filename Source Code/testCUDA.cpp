#include "utils.h" // Assuming this contains necessary utility functions like matrix reader, vector generator, etc.
#include "MatrixDefinitions.h"
#include "CRS/matrixMultivectorProductCRS.h"
#include "CRS/matrixMultivectorProductCRSCUDA.h"
#include "ELLPACK/matrixMultivectorProductELLPACK.h"
#include "ELLPACK/matrixMultivectorProductELLPACKCUDA.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

int main()
{
    std::string filepath = "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/dc1.mtx"; // Path to the matrix file
    int k = 3;                                                                                            // Number of columns in the fat vector
    int testNumber = 10;                                                                                   // Number of iterations for the test 
    int xBlockSize = 64;                                                                                     // Number of threads in the x direction
    int yBlockSize = 16;                                                                                     // Number of threads in the y direction

    try
    {
        SparseMatrixCRS sparseMatrixCRS;
        SparseMatrixELLPACK sparseMatrixELLPACK;
        FatVector fatVector, resultCRS, resultELLPACK, resultELLPACKCUDA, resultCRSCUDA;

        // Read and convert the matrix
        readMatrixMarketFile(filepath, sparseMatrixCRS);
        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK);

        // Generate the dense vector
        generateFatVector(fatVector, sparseMatrixCRS.numCols, k);

        std::cout << "Matrix dimensions: " << sparseMatrixCRS.numRows << "x" << sparseMatrixCRS.numCols << std::endl;

        // CRS Serial
        resultCRS.values.resize(sparseMatrixCRS.numRows * k, 0.0);
        resultCRS.numCols = k;
        resultCRS.numRows = sparseMatrixCRS.numRows;
        matrixMultivectorProductCRS(sparseMatrixCRS, fatVector, resultCRS, testNumber);

        // ELLPACK Serial
        resultELLPACK.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
        resultELLPACK.numCols = k;
        resultELLPACK.numRows = sparseMatrixELLPACK.numRows;
        matrixMultivectorProductELLPACK(sparseMatrixELLPACK, fatVector, resultELLPACK, testNumber);

        // ELLPACK CUDA
        resultELLPACKCUDA.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
        resultELLPACKCUDA.numCols = k;
        resultELLPACKCUDA.numRows = sparseMatrixELLPACK.numRows;
        matrixMultivectorProductELLPACKCUDA(sparseMatrixELLPACK, fatVector, resultELLPACKCUDA, testNumber, xBlockSize, yBlockSize);

        // CRS CUDA
        resultCRSCUDA.values.resize(sparseMatrixCRS.numRows * k, 0.0);
        resultCRSCUDA.numCols = k;
        resultCRSCUDA.numRows = sparseMatrixCRS.numRows;
        matrixMultivectorProductCRSCUDA(sparseMatrixCRS, fatVector, resultCRSCUDA, testNumber, xBlockSize, yBlockSize);

        // Compare the results (assuming areMatricesEqual implementation exists)
        if (areMatricesEqual(resultCRS, resultCRSCUDA, 1e-6))
        {
            std::cout << "CRS results are equal.\n";
        }
        else
        {
            std::cout << "CRS results are not equal.\n";
        } 

        if (areMatricesEqual(resultELLPACK, resultELLPACKCUDA, 1e-6))
        {
            std::cout << "ELLPACK results are equal.\n";
        }
        else
        {
            std::cout << "ELLPACK results are not equal.\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
