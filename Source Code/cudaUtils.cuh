// cuda_utils.cuh
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

/**
 * @brief Check for CUDA errors
 *
 * @param result    The result of a CUDA function call
 * @param func      The name of the CUDA function
 * @param file      The name of the file where the CUDA function was called
 * @param line      The line number in the file where the CUDA function was called
 */
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
                  << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset(); // Reset the GPU to avoid a potential deadlock
        exit(99);          // Exit with an error code
    }
}

/**
 * @brief Atomic add for double precision floating point numbers
 *
 * @param address   The address of the memory location to update
 * @param val      The value to add to the memory location
 */
__device__ inline void atomicAddDouble(double *address, double val)
{
    // Cast the address to a long long int pointer
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    // Convert the double value to a long long int
    unsigned long long int old = *address_as_ull, assumed;
    // Perform the atomic operation
    do
    {
        assumed = old;                                                                                       // Store the old value
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Perform the atomic operation
    } while (assumed != old);                                                                                // Loop until the old value is updated
}

#endif // CUDA_UTILS_CUH
