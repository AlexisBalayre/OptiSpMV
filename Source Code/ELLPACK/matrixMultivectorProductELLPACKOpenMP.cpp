#include "matrixMultivectorProductELLPACKOpenMP.h"

// Assuming SparseMatrixCRS and FatVector structures are defined elsewhere
// SparseMatrixCRS: contains numRows, rowPtr, colIndices, values
// FatVector: vector of vectors or similar structure for dense matrices

double matrixMultivectorProductELLPACKOpenMP(const SparseMatrixELLPACK &sparseMatrix,
                                             const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize)
{
    std::vector<double> times(testNumber); // Store time for each invocation

    // Set the number of threads for OpenMP
    omp_set_num_threads(numThreads);

    for (int n = 0; n < testNumber; ++n)
    {
        // Free the result vector
        result.values.assign(result.values.size(), 0.0);

        double start_time = omp_get_wtime(); // Start timing

        // Parallel computation part with optimizations
        #pragma omp parallel for schedule(dynamic, chunkSize) // Adjust chunk size based on profiling
        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            std::vector<double> local_result(fatVector.numCols, 0.0); // Temporary local result array

            for (int j = 0; j < sparseMatrix.maxNonZerosPerRow; ++j)
            {
                int colIndex = sparseMatrix.colIndices[i * sparseMatrix.maxNonZerosPerRow + j];
                if (colIndex != -1)
                { // Check for valid column index
                    double value = sparseMatrix.values[i * sparseMatrix.maxNonZerosPerRow + j];

                    for (int k = 0; k < fatVector.numCols; ++k)
                    {
                        local_result[k] += value * fatVector.values[colIndex * fatVector.numCols + k];
                    }
                }
            }

            for (int k = 0; k < fatVector.numCols; ++k)
            {
                #pragma omp atomic
                result.values[i * fatVector.numCols + k] += local_result[k];
            }
        }

        double end_time = omp_get_wtime(); // End timing
        times[n] = end_time - start_time;
    }

    // Calculating average time
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    // Optionally, calculate median time
    std::sort(times.begin(), times.end());
    double medianTime = times[testNumber / 2];

    // Calculate performance
    int NZ = sparseMatrix.values.size(); // Number of non-zero entries
    int k = fatVector.numCols;           // Number of columns in matrix X
    double T = avgTime;                  // Average time in seconds
    double FLOPS = (2.0 * NZ * k) / T;
    double GFLOPS = FLOPS / 1e9;

    /* std::cout << "OpenMP Average Time (ELLPACK): " << avgTime << " s" << std::endl;
    std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl;
    */
    return GFLOPS;
}
