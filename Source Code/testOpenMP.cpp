#include "utils.h"
#include "MatrixDefinitions.h"
#include "CRS/matrixMultivectorProductCRS.h"
#include "CRS/matrixMultivectorProductCRSOpenMP.h"
#include "ELLPACK/matrixMultivectorProductELLPACK.h"
#include "ELLPACK/matrixMultivectorProductELLPACKOpenMP.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

int main()
{
    std::string filepath = "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/bcsstk17.mtx";
    int k = 3; // Number of values in the vector
    int testNumber = 10;
    int numThreads = 16; // Number of threads to use
    int chunkSize = 32;  // Chunk size for the OpenMP parallel for

    try
    {
        SparseMatrixCRS sparseMatrixCRS;
        SparseMatrixELLPACK sparseMatrixELLPACK;
        FatVector fatVector, resultCRS, resultELLPACK, resultCRSOpenMP, resultELLPACKOpenMP;

        // Read the matrix from the file and prepare the structures
        readMatrixMarketFile(filepath, sparseMatrixCRS);

        std::cout << "Matrix dimensions: " << sparseMatrixCRS.numRows << "x" << sparseMatrixCRS.numCols << std::endl;
        std::cout << "Number of non-zero elements: " << sparseMatrixCRS.values.size() << std::endl;

        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK);
        std::cout << "ELLPACK matrix dimensions: " << sparseMatrixELLPACK.numRows << "x" << sparseMatrixELLPACK.numCols << std::endl;

        generateFatVector(fatVector, sparseMatrixCRS.numCols, k);

        // Perform the matrix-vector multiplication (CRS - Serial)
        resultCRS.values.resize(sparseMatrixCRS.numRows * k, 0.0);
        resultCRS.numCols = k;
        resultCRS.numRows = sparseMatrixCRS.numRows;
        matrixMultivectorProductCRS(sparseMatrixCRS, fatVector, resultCRS, testNumber);

        // Perform the matrix-vector multiplication (ELLPACK - Serial)
        resultELLPACK.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
        resultELLPACK.numCols = k;
        resultELLPACK.numRows = sparseMatrixELLPACK.numRows;
        matrixMultivectorProductELLPACK(sparseMatrixELLPACK, fatVector, resultELLPACK, testNumber);

        // Perform the matrix-vector multiplication (CRS - OpenMP)
        resultCRSOpenMP.values.resize(sparseMatrixCRS.numRows * k, 0.0);
        resultCRSOpenMP.numCols = k;
        resultCRSOpenMP.numRows = sparseMatrixCRS.numRows;
        matrixMultivectorProductCRSOpenMP(sparseMatrixCRS, fatVector, resultCRSOpenMP, testNumber, numThreads, chunkSize);

        // Perform the matrix-vector multiplication (ELLPACK - OpenMP)
        resultELLPACKOpenMP.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
        resultELLPACKOpenMP.numCols = k;
        resultELLPACKOpenMP.numRows = sparseMatrixELLPACK.numRows;
        matrixMultivectorProductELLPACKOpenMP(sparseMatrixELLPACK, fatVector, resultELLPACKOpenMP, testNumber, numThreads, chunkSize);

        // Compare the results
        if (areMatricesEqual(resultCRS, resultCRSOpenMP, 1e-6))
        {
            std::cout << "The CRS results are equal." << std::endl;
        }
        else
        {
            std::cout << "The CRS results are not equal." << std::endl;
        }

        if (areMatricesEqual(resultELLPACK, resultELLPACKOpenMP, 1e-6))
        {
            std::cout << "The ELLPACK results are equal." << std::endl;
        }
        else
        {
            std::cout << "The ELLPACK results are not equal." << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
