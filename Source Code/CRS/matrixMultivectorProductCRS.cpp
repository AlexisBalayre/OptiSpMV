#include "matrixMultivectorProductCRS.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param fatVector Fat vector
 * @param vecCols  Number of columns in the dense vector
 * @return FatVector  Result of the multiplication
 */

double matrixMultivectorProductCRS(const SparseMatrixCRS &sparseMatrix,
                                   const FatVector &fatVector, FatVector &result, const int testNumber)
{
    std::vector<double> times(testNumber); // To store time taken for each invocation in milliseconds

    for (int i = 0; i < testNumber; ++i)
    {
        // Free the result vector
        result.values.assign(result.values.size(), 0.0);
        
        auto start = std::chrono::high_resolution_clock::now(); // Start timing

        int numRows = sparseMatrix.numRows; // Number of rows in the sparse matrix
        int vecCols = fatVector.numCols;    // Number of columns in the dense vector

        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < numRows; ++i)
        {
            // Iterate over the non-zero elements in the current row
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int colIndex = sparseMatrix.colIndices[j]; // Column index in the sparse matrix
                double value = sparseMatrix.values[j];     // Value of the non-zero element

                // Accumulate the product into the result vector
                for (int k = 0; k < vecCols; ++k)
                {
                    // Calculate the flat index for the result vector
                    int resultIndex = i * vecCols + k;
                    // Calculate the flat index for the dense vector (fatVector)
                    int vectorIndex = colIndex * vecCols + k;

                    result.values[resultIndex] += value * fatVector.values[vectorIndex];
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    // Calculating average time per kernel invocation
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    // Optionally, calculate median time
    std::sort(times.begin(), times.end());
    double medianTime = times[testNumber / 2];

    // Calculate performance
    int NZ = sparseMatrix.values.size(); // Number of non-zero entries
    int k = fatVector.numCols;           // Number of columns in matrix X
    double T = avgTime / 1000.0;         // Average time in seconds
    double FLOPS = (2.0 * NZ * k) / T;
    double GFLOPS = FLOPS / 1e9;

    /* std::cout << "Average Time (ms): " << avgTime << std::endl;
    std::cout << "GFLOPS (CRS): " << GFLOPS << std::endl; */

    return GFLOPS;
}
