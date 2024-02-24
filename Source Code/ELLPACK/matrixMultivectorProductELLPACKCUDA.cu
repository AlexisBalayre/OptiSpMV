#include "matrixMultivectorProductELLPACKCUDA.h"
#include "../cudaUtils.cuh"

/**
 * @brief Kernel function to multiply sparse matrix with a fat vector using ELLPACK format on CUDA
 *
 * @param values Non-zero values of the sparse matrix
 * @param colIndices Column indices of the non-zero values
 * @param vector Fat vector
 * @param result Result vector
 * @param numRows Number of rows in the sparse matrix
 * @param maxNonZerosPerRow Maximum number of non-zero values per row
 * @param vecCols Number of columns in the fat vector
 */
__global__ void spmvEllpackKernel(const double *values, const int *colIndices, const double *vector, double *result, int numRows, int maxNonZerosPerRow, int vecCols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Row index
    int col = threadIdx.x;                           // Column index
    // Check for valid indices
    if (row < numRows && col < vecCols)
    {
        double sum = 0.0; // Initialize sum
        // Iterate over non-zero values of the row
        for (int i = 0; i < maxNonZerosPerRow; ++i)
        {
            int idx = row * maxNonZerosPerRow + i; // Index of the non-zero value
            int colIndex = colIndices[idx];        // Column index of the non-zero value
            // Check for valid column index
            if (colIndex != -1)
            {
                sum += values[idx] * vector[colIndex * vecCols + col]; // Multiply and add to sum
            }
        }
        atomicAddDouble(&result[row * vecCols + col], sum); // Add sum to the result
    }
}

/**
 * @brief Multiply sparse matrix with a fat vector using ELLPACK format on CUDA
 *
 * @param sparseMatrix Sparse matrix in ELLPACK format
 * @param fatVector Fat vector
 * @param result Result vector
 * @param testNumber Number of times to run the operation
 * @param xBlockSize X-dimension block size
 * @param yBlockSize Y-dimension block size
 * @return double Performance in GFLOPS
 */
double matrixMultivectorProductELLPACKCUDA(const SparseMatrixELLPACK &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize)
{
    std::vector<float> times(testNumber); // Store time for each invocation

    const dim3 BLOCK_DIM(xBlockSize, yBlockSize); // Block dimensions

    cudaEvent_t start, stop; // CUDA events to measure time
    cudaEventCreate(&start); // Create start event
    cudaEventCreate(&stop);  // Create stop event

    // Memory allocations and transfers
    double *d_values, *d_fatVector, *d_result;                    // Device memory pointers for values, fat vector and result
    int *d_colIndices;                                            // Device memory pointer for column indices
    size_t valuesSize = sparseMatrix.values.size();               // Size of non-zero values
    size_t colIndicesSize = sparseMatrix.colIndices.size();       // Size of column indices
    size_t fatVectorSize = fatVector.values.size();               // Size of fat vector
    size_t resultSize = sparseMatrix.numRows * fatVector.numCols; // Size of result vector

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_values, valuesSize * sizeof(double)));                                                             // Allocate memory for non-zero values
    checkCudaErrors(cudaMalloc(&d_colIndices, colIndicesSize * sizeof(int)));                                                        // Allocate memory for column indices
    checkCudaErrors(cudaMalloc(&d_fatVector, fatVectorSize * sizeof(double)));                                                       // Allocate memory for fat vector
    checkCudaErrors(cudaMalloc(&d_result, resultSize * sizeof(double)));                                                             // Allocate memory for result
    checkCudaErrors(cudaMemcpy(d_values, sparseMatrix.values.data(), valuesSize * sizeof(double), cudaMemcpyHostToDevice));          // Transfer non-zero values to device
    checkCudaErrors(cudaMemcpy(d_colIndices, sparseMatrix.colIndices.data(), colIndicesSize * sizeof(int), cudaMemcpyHostToDevice)); // Transfer column indices to device
    checkCudaErrors(cudaMemcpy(d_fatVector, fatVector.values.data(), fatVectorSize * sizeof(double), cudaMemcpyHostToDevice));       // Transfer fat vector to device
    checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));                                                           // Initialize result memory

    // Run the operation multiple times
    for (int i = 0; i < testNumber; ++i)
    {
        checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double))); // Reset result memory

        // Perform the matrix-vector multiplication
        cudaEventRecord(start);                                                                                                                                                               // Start measuring time
        dim3 GRID_DIM((sparseMatrix.numRows + yBlockSize - 1) / yBlockSize, 1, 1);                                                                                                            // Grid dimensions
        size_t sharedMemorySize = fatVector.numCols * sizeof(double);                                                                                                                         // Allocate shared memory dynamically
        spmvEllpackKernel<<<GRID_DIM, BLOCK_DIM, sharedMemorySize>>>(d_values, d_colIndices, d_fatVector, d_result, sparseMatrix.numRows, sparseMatrix.maxNonZerosPerRow, fatVector.numCols); // Kernel launch
        checkCudaErrors(cudaDeviceSynchronize());                                                                                                                                             // Wait for kernel to finish
        checkCudaErrors(cudaGetLastError());                                                                                                                                                  // Check for errors
        checkCudaErrors(cudaMemcpy(result.values.data(), d_result, resultSize * sizeof(double), cudaMemcpyDeviceToHost));                                                                     // Transfer result to host
        cudaEventRecord(stop);                                                                                                                                                                // Stop measuring time
        checkCudaErrors(cudaEventSynchronize(stop));                                                                                                                                          // Wait for stop event

        // Calculate time execution
        float milliseconds = 0;                                            // Time in milliseconds
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop)); // Calculate time
        times[i] = milliseconds;                                           // Store time
    }

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_colIndices);
    cudaFree(d_fatVector);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate performance
    float avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0f) / times.size(); // Average time in milliseconds
    int NZ = sparseMatrix.values.size();                                                // Number of non-zero entries
    int k = fatVector.numCols;                                                          // Number of columns in matrix X
    float T = avgTimeMs / 1000.0f;                                                      // Convert to seconds
    double FLOPS = (2.0 * NZ * k) / T;                                                  // Calculate FLOPS
    double GFLOPS = FLOPS / 1e9;                                                        // Convert to GFLOPS

    // DISPLAY PERFORMANCE (FOR DEBUGGING)
    // std::cout << "Average CUDA operation time: " << avgTimeMs << " ms" << std::endl;
    // std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl;

    return GFLOPS;
}