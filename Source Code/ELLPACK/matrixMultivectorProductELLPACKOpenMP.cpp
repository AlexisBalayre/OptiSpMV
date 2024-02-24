#include "matrixMultivectorProductELLPACKOpenMP.h"

/**
 * @brief Perform the matrix-vector multiplication for an ELLPACK matrix using OpenMP
 *
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Dense vector
 * @param result Pointer to the result vector
 * @param testNumber Number of iterations for the performance measurement
 * @param numThreads Number of threads to use
 * @param chunkSize Chunk size for the OpenMP parallel for
 * @return double GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductELLPACKOpenMP(const SparseMatrixELLPACK &sparseMatrix,
                                             const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize)
{
    std::vector<double> times(testNumber); // Store time for each invocation

    omp_set_num_threads(numThreads); // Set the number of threads for OpenMP

    // Perform testNumber iterations
    for (int n = 0; n < testNumber; ++n)
    {
        result.values.assign(result.values.size(), 0.0); // Free the result vector

        double start_time = omp_get_wtime(); // Start timing

        // Parallel computation part with optimizations
        #pragma omp parallel for schedule(dynamic, chunkSize) // Parallelize the loop using OpenMP with dynamic scheduling and a chunk size
        // Iterate over the rows of the sparse matrix
        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            std::vector<double> local_result(fatVector.numCols, 0.0); // Temporary local result array
            // Iterate over each possible non-zero element in the row, up to the maximum number per row
            for (int j = 0; j < sparseMatrix.maxNonZerosPerRow; ++j)
            {
                int colIndex = sparseMatrix.colIndices[i * sparseMatrix.maxNonZerosPerRow + j]; // Column index in the sparse matrix
                // If the column index is -1, it indicates padding and should be ignored
                if (colIndex != -1)
                {
                    double value = sparseMatrix.values[i * sparseMatrix.maxNonZerosPerRow + j]; // Value of the non-zero element
                    // Iterate over the columns of the dense vector
                    for (int k = 0; k < fatVector.numCols; ++k)
                    {
                        local_result[k] += value * fatVector.values[colIndex * fatVector.numCols + k]; // Compute the result
                    }
                }
            }
            // Combine local results
            for (int k = 0; k < fatVector.numCols; ++k)
            {
                #pragma omp atomic                                                           // Atomic operation
                result.values[i * fatVector.numCols + k] += local_result[k]; // Accumulate the local result into the global result
            }
        }

        double end_time = omp_get_wtime(); // End timing
        times[n] = end_time - start_time;  // Store the duration
    }

    // Calculate performance
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size(); // Calculate average time
    int NZ = sparseMatrix.values.size();                                              // Number of non-zero entries
    int k = fatVector.numCols;                                                        // Number of columns in matrix X
    double T = avgTime;                                                               // Average time in seconds
    double FLOPS = (2.0 * NZ * k) / T;                                                // Number of floating point operations
    double GFLOPS = FLOPS / 1e9;                                                      // Performance in GFLOPS

    // DISPLAY RESULTS (FOR DEBUGGING)
    // std::cout << "OpenMP Average Time (ELLPACK): " << avgTime << " s" << std::endl;
    // std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl;

    return GFLOPS;
}
