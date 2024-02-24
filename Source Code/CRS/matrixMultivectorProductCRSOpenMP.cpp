#include "matrixMultivectorProductCRSOpenMP.h"

/**
 * @brief Perform the matrix-vector multiplication in the CRS format using OpenMP
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector  Dense vector
 * @param result  Pointer to the result vector
 * @param testNumber  Number of iterations for the performance measurement
 * @param numThreads  Number of threads to use
 * @param chunkSize  Chunk size for the OpenMP parallel for
 * @return double  GFLOPS Performance of the kernel in GFLOPS
 */
double matrixMultivectorProductCRSOpenMP(const SparseMatrixCRS &sparseMatrix,
                                         const FatVector &fatVector, FatVector &result, const int testNumber, const int numThreads, const int chunkSize)
{
    std::vector<double> times(testNumber); // Store time for each invocation

    omp_set_num_threads(numThreads); // Set the number of threads for OpenMP

    // Perform testNumber iterations
    for (int n = 0; n < testNumber; ++n)
    {
        result.values.assign(result.values.size(), 0.0); // Free the result vector

        double start_time = omp_get_wtime(); // Start timing

        // Perform the matrix-vector multiplication
        #pragma omp parallel for schedule(dynamic, chunkSize) // Parallelize the loop using OpenMP with dynamic scheduling and a chunk size
        for (int i = 0; i < sparseMatrix.numRows; ++i)
        {
            double local_result[fatVector.numCols] = {0}; // Temporary local result array
            // Iterate over the non-zero elements in the current row
            for (int j = sparseMatrix.rowPtr[i]; j < sparseMatrix.rowPtr[i + 1]; ++j)
            {
                int colIndex = sparseMatrix.colIndices[j]; // Column index in the sparse matrix
                double value = sparseMatrix.values[j];     // Value of the non-zero element
                // Accumulate the product into the result vector
                for (int k = 0; k < fatVector.numCols; ++k)
                {
                    local_result[k] += value * fatVector.values[colIndex * fatVector.numCols + k]; // Perform the multiplication and accumulation
                }
            }
            // Combine local results
            for (int k = 0; k < fatVector.numCols; ++k)
            { // Combine local results
                #pragma omp atomic // Atomic operation
                result.values[i * fatVector.numCols + k] += local_result[k]; // Accumulate the local result into the global result
            }
        }

        double end_time = omp_get_wtime(); // End timing
        times[n] = end_time - start_time;  // Store the duration
    }

    // Calculate performance
    double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    int NZ = sparseMatrix.values.size(); // Number of non-zero entries
    int k = fatVector.numCols;           // Number of columns in matrix X
    double T = avgTime;                  // Average time in seconds
    double FLOPS = (2.0 * NZ * k) / T;  // Number of floating point operations
    double GFLOPS = FLOPS / 1e9;        // Performance in GFLOPS

    // DISPLAY RESULTS (FOR DEBUGGING)
    // std::cout << "OpenMP Average Time (CRS): " << avgTime << " s" << std::endl;
    // std::cout << "Performance (CRS): " << GFLOPS << " GFLOPS" << std::endl;

    return GFLOPS;
}
