#include "matrixMultivectorProductCRS.h"

/**
 * @brief Perform the matrix-vector multiplication in the CRS format
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductCRS(const SparseMatrixCRS &sparseMatrix,
                                   const FatVector &fatVector, FatVector &result, const int testNumber)
{
    std::vector<double> times(testNumber); // Vector for storing times of individual iterations

    // Perform testNumber iterations
    for (int i = 0; i < testNumber; ++i)
    {
        result.values.assign(result.values.size(), 0.0); // Free the result vector

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
                    int resultIndex = i * vecCols + k;                                   // Calculate the flat index for the result vector
                    int vectorIndex = colIndex * vecCols + k;                            // Calculate the flat index for the dense vector
                    result.values[resultIndex] += value * fatVector.values[vectorIndex]; // Perform the multiplication and accumulationw
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();             // End timing
        std::chrono::duration<double, std::milli> duration = end - start; // Calculate the duration
        times.push_back(duration.count());                                // Store the duration
    }

    // Calculate performance
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size(); // Calculate average time
    int NZ = sparseMatrix.values.size();                                              // Number of non-zero entries
    int k = fatVector.numCols;                                                        // Number of columns in matrix X
    double T = avgTime / 1000.0;                                                      // Average time in seconds
    double FLOPS = (2.0 * NZ * k) / T;                                                // Number of floating point operations
    double GFLOPS = FLOPS / 1e9;                                                      // Performance in GFLOPS

    // DISPLAY RESULTS (FOR DEBUGGING)
    // std::cout << "Average Time (ms): " << avgTime << std::endl;
    // std::cout << "GFLOPS (CRS): " << GFLOPS << std::endl; 

    return GFLOPS;
}
