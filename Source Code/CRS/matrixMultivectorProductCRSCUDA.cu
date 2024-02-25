#include "matrixMultivectorProductCRSCUDA.h"
#include "../cudaUtils.cuh"

/**
 * @brief Kernel for sparse matrix-vector product in CRS format
 *
 * @param values Non-zero values of the matrix
 * @param colIndices Column indices of the non-zero values
 * @param rowPtr Row pointer of the matrix
 * @param vector  Input vector
 * @param result Output vector
 * @param numRows Number of rows in the matrix
 * @param vecCols Number of columns in the input vector
 */
__global__ void spmvCrsKernel(const double *values, const int *colIndices, const int *rowPtr, const double *vector, double *result, int numRows, int vecCols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y; // Row index
    int col = threadIdx.x;                           // Column index
    // Check if the thread is within the matrix dimensions
    if (row < numRows && col < vecCols)
    {
        double sum = 0.0;             // Initialize the sum
        int rowStart = rowPtr[row];   // Start index of the row
        int rowEnd = rowPtr[row + 1]; // End index of the row
        // Loop over the non-zero values of the row
        for (int i = rowStart; i < rowEnd; ++i)
        {
            int colIndex = colIndices[i];                        // Column index of the non-zero value
            sum += values[i] * vector[colIndex * vecCols + col]; // Multiply and accumulate
        }
        atomicAddDouble(&result[row * vecCols + col], sum); // Store the result
    }
}

/**
 * @brief Perform matrix-vector product using CUDA
 *
 * @param sparseMatrix Sparse matrix in CRS format
 * @param fatVector Input vector
 * @param result Output vector
 * @param testNumber Number of invocations
 * @param xBlockSize X-dimension of the block
 * @param yBlockSize Y-dimension of the block
 * @return double Performance in GFLOPS
 */
double matrixMultivectorProductCRSCUDA(const SparseMatrixCRS &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize)
{
    std::vector<float> times(testNumber); // Store time for each invocation

    const dim3 BLOCK_DIM(xBlockSize, yBlockSize); // Block dimensions

    cudaEvent_t start, stop; // CUDA events for timing
    cudaEventCreate(&start); // Start event
    cudaEventCreate(&stop);  // Stop event

    // Device memory
    double *d_values, *d_fatVector, *d_result;                    // Device pointers for the values, input vector and result
    int *d_colIndices, *d_rowPtr;                                 // Device pointers for the column indices and row pointer
    size_t valuesSize = sparseMatrix.values.size();               // Non-zero values size
    size_t colIndicesSize = sparseMatrix.colIndices.size();       // Column indices size
    size_t rowPtrSize = sparseMatrix.rowPtr.size();               // Row pointer size
    size_t fatVectorSize = fatVector.values.size();               // Input vector size
    size_t resultSize = sparseMatrix.numRows * fatVector.numCols; // Result size

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_values, valuesSize * sizeof(double)));                                                             // Allocate memory for the non-zero values
    checkCudaErrors(cudaMalloc(&d_colIndices, colIndicesSize * sizeof(int)));                                                        // Allocate memory for the column indices
    checkCudaErrors(cudaMalloc(&d_rowPtr, rowPtrSize * sizeof(int)));                                                                // Allocate memory for the row pointer
    checkCudaErrors(cudaMalloc(&d_fatVector, fatVectorSize * sizeof(double)));                                                       // Allocate memory for the input vector
    checkCudaErrors(cudaMalloc(&d_result, resultSize * sizeof(double)));                                                             // Allocate memory for the result
    checkCudaErrors(cudaMemcpy(d_values, sparseMatrix.values.data(), valuesSize * sizeof(double), cudaMemcpyHostToDevice));          // Copy non-zero values to device
    checkCudaErrors(cudaMemcpy(d_colIndices, sparseMatrix.colIndices.data(), colIndicesSize * sizeof(int), cudaMemcpyHostToDevice)); // Copy column indices to device
    checkCudaErrors(cudaMemcpy(d_rowPtr, sparseMatrix.rowPtr.data(), rowPtrSize * sizeof(int), cudaMemcpyHostToDevice));             // Copy row pointer to device
    checkCudaErrors(cudaMemcpy(d_fatVector, fatVector.values.data(), fatVectorSize * sizeof(double), cudaMemcpyHostToDevice));       // Copy input vector to device
    checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));

    // Perform the test multiple times
    for (int i = 0; i < testNumber; ++i)
    {
        checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double))); // Reset result to zero

        // Perform the sparse matrix-vector product
        cudaEventRecord(start);                                                                                                                   // Start timing
        dim3 GRID_DIM((sparseMatrix.numRows + yBlockSize - 1) / yBlockSize, 1, 1);                                                                // Grid dimensions
        spmvCrsKernel<<<GRID_DIM, BLOCK_DIM>>>(d_values, d_colIndices, d_rowPtr, d_fatVector, d_result, sparseMatrix.numRows, fatVector.numCols); // Kernel launch
        checkCudaErrors(cudaDeviceSynchronize());                                                                                                 // Wait for the kernel to finish
        checkCudaErrors(cudaGetLastError());                                                                                                      // Check for errors
        checkCudaErrors(cudaMemcpy(result.values.data(), d_result, resultSize * sizeof(double), cudaMemcpyDeviceToHost));                         // Copy the result back to host
        cudaEventRecord(stop);                                                                                                                    // Stop timing
        checkCudaErrors(cudaEventSynchronize(stop));                                                                                              // Wait for the stop event to be recorded

        // Calculate execution time
        float milliseconds = 0;                                            // Time in milliseconds
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop)); // Calculate the elapsed time
        times[i] = milliseconds;                                           // Store the time
    }

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_colIndices);
    cudaFree(d_rowPtr);
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


    // DISPLAY PERFORMANCE (FOR DEBUGGING PURPOSES)
    // std::cout << "Average CUDA operation time: " << avgTimeMs << " ms" << std::endl;
    // std::cout << "Performance (CRS): " << GFLOPS << " GFLOPS" << std::endl; 

    return GFLOPS;
}