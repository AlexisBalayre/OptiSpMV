#include "matrixMultivectorProductELLPACK.h" // Include guard for the ELLPACK versio

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm for ELLPACK format
 *
 * @param sparseMatrix  Sparse matrix in ELLPACK format
 * @param fatVector     Dense vector (2D vector)
 * @param vecCols       Number of columns in the dense vector
 * @return FatVector    Result of the multiplication
 */
double matrixMultivectorProductELLPACK(const SparseMatrixELLPACK &sparseMatrix,
                                       const FatVector &fatVector, FatVector &result, const int testNumber)
{

    std::vector<double> times(testNumber); // Store time for each invocation in milliseconds

    for (int n = 0; n < testNumber; ++n)
    {
        // Free the result vector
        result.values.assign(result.values.size(), 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();

        int numRows = sparseMatrix.numRows; // Number of rows in the sparse matrix
        int vecCols = fatVector.numCols;    // Number of columns in the dense vector

        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < numRows; ++i)
        {
            // Iterate over each possible non-zero element in the row, up to the maximum number per row
            for (int j = 0; j < sparseMatrix.maxNonZerosPerRow; ++j)
            {
                int colIndex = sparseMatrix.colIndices[i * sparseMatrix.maxNonZerosPerRow + j];
                double value = sparseMatrix.values[i * sparseMatrix.maxNonZerosPerRow + j];
                // If the column index is -1, it indicates padding and should be ignored
                if (colIndex != -1)
                {
                    // Iterate over the columns of the dense vector
                    for (int k = 0; k < vecCols; ++k)
                    {
                        int resultIndex = i * vecCols + k;
                        int vectorIndex = colIndex * vecCols + k;
                        result.values[resultIndex] += value * fatVector.values[vectorIndex]; // Compute the result
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        times[n] = duration.count();
    }

    // Calculate average and median times
    double avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::sort(times.begin(), times.end());
    double medianTimeMs = times[testNumber / 2];

    // Calculate performance
    int NZ = sparseMatrix.numRows * sparseMatrix.maxNonZerosPerRow; // Total potential non-zeros, adjustments for actual non-zeros might be needed
    int k = fatVector.numCols;
    double avgTimeSec = avgTimeMs / 1000.0; // Convert milliseconds to seconds for calculation
    double FLOPS = (2.0 * NZ * k) / avgTimeSec;
    double GFLOPS = FLOPS / 1e9;

    /* std::cout << "Average Time: " << avgTimeMs << " ms" << std::endl;
    std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl; */

    return GFLOPS;
}
