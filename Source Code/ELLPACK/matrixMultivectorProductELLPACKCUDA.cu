#include "matrixMultivectorProductELLPACKCUDA.h"
#include "../cudaUtils.cuh"

__global__ void spmvEllpackKernel(const double *values, const int *colIndices, const double *vector, double *result, int numRows, int maxNonZerosPerRow, int vecCols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    if (row < numRows && col < vecCols)
    {
        double sum = 0.0;
        for (int i = 0; i < maxNonZerosPerRow; ++i)
        {
            int idx = row * maxNonZerosPerRow + i;
            int colIndex = colIndices[idx];
            if (colIndex != -1)
            { // Check for valid column index
                sum += values[idx] * vector[colIndex * vecCols + col];
            }
        }
        atomicAddDouble(&result[row * vecCols + col], sum);
    }
}

// Host function to multiply sparse matrix with a fat vector using ELLPACK format on CUDA, refined to match the provided definitions
double matrixMultivectorProductELLPACKCUDA(const SparseMatrixELLPACK &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize)
{
    std::vector<float> times(testNumber); // Store time for each invocation

    const dim3 BLOCK_DIM(xBlockSize, yBlockSize); // Block dimensions

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Memory allocations and transfers
    double *d_values, *d_fatVector, *d_result;
    int *d_colIndices;
    size_t valuesSize = sparseMatrix.values.size();
    size_t colIndicesSize = sparseMatrix.colIndices.size();
    size_t fatVectorSize = fatVector.values.size();
    size_t resultSize = sparseMatrix.numRows * fatVector.numCols;

    checkCudaErrors(cudaMalloc(&d_values, valuesSize * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_colIndices, colIndicesSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_fatVector, fatVectorSize * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_result, resultSize * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_values, sparseMatrix.values.data(), valuesSize * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colIndices, sparseMatrix.colIndices.data(), colIndicesSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fatVector, fatVector.values.data(), fatVectorSize * sizeof(double), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));

    for (int i = 0; i < testNumber; ++i)
    {
        // Free result memory
        checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));
        
        // Kernel launch
        cudaEventRecord(start);
        dim3 GRID_DIM((sparseMatrix.numRows + yBlockSize - 1) / yBlockSize, 1, 1);
        size_t sharedMemorySize = fatVector.numCols * sizeof(double); // Allocate shared memory dynamically
        spmvEllpackKernel<<<GRID_DIM, BLOCK_DIM, sharedMemorySize>>>(d_values, d_colIndices, d_fatVector, d_result, sparseMatrix.numRows, sparseMatrix.maxNonZerosPerRow, fatVector.numCols);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(result.values.data(), d_result, resultSize * sizeof(double), cudaMemcpyDeviceToHost));

        cudaEventRecord(stop);
        checkCudaErrors(cudaEventSynchronize(stop));

        float milliseconds = 0;
        checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
        times[i] = milliseconds;
    }

    // Cleanup
    cudaFree(d_values);
    cudaFree(d_colIndices);
    cudaFree(d_fatVector);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate average time
    float avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    // Calculate GFLOPS
    int NZ = sparseMatrix.values.size(); // Number of non-zero entries
    int k = fatVector.numCols;           // Number of columns in matrix X
    float T = avgTimeMs / 1000.0f;       // Convert to seconds
    double FLOPS = (2.0 * NZ * k) / T;
    double GFLOPS = FLOPS / 1e9;

    /* std::cout << "Average CUDA operation time: " << avgTimeMs << " ms" << std::endl;
    std::cout << "Performance (ELLPACK): " << GFLOPS << " GFLOPS" << std::endl; */

    return GFLOPS;
}