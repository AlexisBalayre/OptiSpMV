// cuda_utils.cuh
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// Host-side error checking function
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
                  << file << ":" << line << " '" << func << "' \n";
        // Consider handling CUDA errors in a way that's appropriate for your application
        cudaDeviceReset(); // Note: cudaDeviceReset will destroy all allocations and reset all devices
        exit(99); // or handle the error without exiting if appropriate
    }
}

// Device-side atomic addition for double precision
__device__ inline void atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}

#endif // CUDA_UTILS_CUH
