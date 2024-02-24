#include "matrixMultivectorProductELLPACK.h" // Include guard for the ELLPACK versio

/**
 * @brief Perform the matrix-vector multiplication for an ELLPACK matrix
 *
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductELLPACK(const SparseMatrixELLPACK &sparseMatrix,
                                       const FatVector &fatVector, FatVector &result, const int testNumber)
{

    std::vector<double> times(testNumber); // Store time for each invocation in milliseconds

    // Perform testNumber iterations
    for (int n = 0; n < testNumber; ++n)
    {
        result.values.assign(result.values.size(), 0.0); // Free the result vector

        auto start = std::chrono::high_resolution_clock::now(); // Start timing

        int numRows = sparseMatrix.numRows; // Number of rows in the sparse matrix
        int vecCols = fatVector.numCols;    // Number of columns in the dense vector

        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < numRows; ++i)
        {
            // Iterate over each possible non-zero element in the row, up to the maximum number per row
            for (int j = 0; j < sparseMatrix.maxNonZerosPerRow; ++j)
            {
                int colIndex = sparseMatrix.colIndices[i * sparseMatrix.maxNonZerosPerRow + j]; // Column index in the sparse matrix
                double value = sparseMatrix.values[i * sparseMatrix.maxNonZerosPerRow + j];     // Value of the non-zero element
                // If the column index is -1, it indicates padding and should be ignored
                if (colIndex != -1)
                {
                    // Iterate over the columns of the dense vector
                    for (int k = 0; k < vecCols; ++k)
                    {
                        int resultIndex = i * vecCols + k;                                   // Calculate the flat index for the result vector
                        int vectorIndex = colIndex * vecCols + k;                            // Calculate the flat index for the dense vector
                        result.values[resultIndex] += value * fatVector.values[vectorIndex]; // Compute the result
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();             // End timing
        std::chrono::duration<double, std::milli> duration = end - start; // Calculate the duration
        times[n] = duration.count();                                      // Store the duration
    }

    // Calculate performance
    double avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size(); // Calculate average time in milliseconds
    int NZ = sparseMatrix.numRows * sparseMatrix.maxNonZerosPerRow;                     // Total potential non-zeros, adjustments for actual non-zeros might be needed
    int k = fatVector.numCols;                                                          // Number of columns in matrix X
    double avgTimeSec = avgTimeMs / 1000.0;                                             // Convert milliseconds to seconds for calculation
    double FLOPS = (2.0 * NZ * k) / avgTimeSec;                                         // Number of floating point operations
    double GFLOPS = FLOPS / 1e9;                                                        // Performance in GFLOPS

    // DISPLAY RESULTS (FOR DEBUGGING)
    // std::cout << "Average Time: " << avgTimeMs << " ms" << std::endl;
    // std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl;

    return GFLOPS;
}
