#include "matrixMultivectorProductCRSCUDA.h"
#include "../cudaUtils.cuh"

__global__ void spmvCrsKernel(const double *values, const int *colIndices, const int *rowPtr, const double *vector, double *result, int numRows, int vecCols)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int col = threadIdx.x;
    if (row < numRows && col < vecCols)
    {
        double sum = 0.0;
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];
        for (int i = rowStart; i < rowEnd; ++i)
        {
            int colIndex = colIndices[i];
            sum += values[i] * vector[colIndex * vecCols + col];
        }
        atomicAddDouble(&result[row * vecCols + col], sum);
    }
}

double matrixMultivectorProductCRSCUDA(const SparseMatrixCRS &sparseMatrix, const FatVector &fatVector, FatVector &result, const int testNumber, const int xBlockSize, const int yBlockSize)
{
    std::vector<float> times(testNumber); // Store time for each invocation

    const dim3 BLOCK_DIM(xBlockSize, yBlockSize); // Block dimensions

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *d_values, *d_fatVector, *d_result;
    int *d_colIndices, *d_rowPtr;
    size_t valuesSize = sparseMatrix.values.size();
    size_t colIndicesSize = sparseMatrix.colIndices.size();
    size_t rowPtrSize = sparseMatrix.rowPtr.size();
    size_t fatVectorSize = fatVector.values.size();
    size_t resultSize = sparseMatrix.numRows * fatVector.numCols;

    checkCudaErrors(cudaMalloc(&d_values, valuesSize * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_colIndices, colIndicesSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_rowPtr, rowPtrSize * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_fatVector, fatVectorSize * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_result, resultSize * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_values, sparseMatrix.values.data(), valuesSize * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colIndices, sparseMatrix.colIndices.data(), colIndicesSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rowPtr, sparseMatrix.rowPtr.data(), rowPtrSize * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fatVector, fatVector.values.data(), fatVectorSize * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));

    for (int i = 0; i < testNumber; ++i)
    {
        // Free result
        checkCudaErrors(cudaMemset(d_result, 0, resultSize * sizeof(double)));

        // Kernel launch
        cudaEventRecord(start);
        dim3 GRID_DIM((sparseMatrix.numRows + yBlockSize - 1) / yBlockSize, 1, 1);
        spmvCrsKernel<<<GRID_DIM, BLOCK_DIM>>>(d_values, d_colIndices, d_rowPtr, d_fatVector, d_result, sparseMatrix.numRows, fatVector.numCols);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        // Copy result back to host
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
    cudaFree(d_rowPtr);
    cudaFree(d_fatVector);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calculate average time
    float avgTimeMs = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();

    // Calculate GFLOPS
    int NZ = sparseMatrix.values.size(); // Number of non-zero entries
    int k = fatVector.numCols;           // Number of columns in matrix X
    float GFLOPS = (2.e-6 * NZ * k) / avgTimeMs;

    /* std::cout << "Average CUDA operation time: " << avgTimeMs << " ms" << std::endl;
    std::cout << "Performance (CRS): " << GFLOPS << " GFLOPS" << std::endl; */

    return GFLOPS;
}