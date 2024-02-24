#include "utils.h" // Assuming this contains necessary utility functions like matrix reader, vector generator, etc.
#include "MatrixDefinitions.h"
#include "CRS/matrixMultivectorProductCRS.h"
#include "CRS/matrixMultivectorProductCRSCUDA.h"
#include "ELLPACK/matrixMultivectorProductELLPACK.h"
#include "ELLPACK/matrixMultivectorProductELLPACKCUDA.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

int main()
{
    // List of filepaths to the matrices
    std::vector<std::string> filepaths = {
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/cop20k_A.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/adder_dcop_32.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/bcsstk17.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/af_1_k101.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/af23560.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/amazon0302.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/cant.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/cavity10.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/cage4.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/dc1.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/Cube_Coup_dt0.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/FEM_3D_thermal1.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/mac_econ_fwd500.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/mcfe.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/mhd4800a.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/ML_Laplace.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/nlpkkt80.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/olafu.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/PR02R.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/raefsky2.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/rdist2.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/thermal1.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/thermal2.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/thermomech_TK.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/west2021.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/webbase-1M.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/lung2.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/olm1000.mtx",
        "/mnt/beegfs/home/s425500/small_scale/Assignment/sparse-matrix/roadNet-PA.mtx"};
    std::vector<int> ks = {1, 2, 3, 6};             // Column number of the fat vector
    std::vector<int> xbdOptions = {8, 16, 32, 64};  // Block size options for the x-axis
    std::vector<int> ybdOptions = {1, 2, 4, 8, 16}; // Block size options for the y-axis
    int iterationsNum = 20;                         // Number of iterations for the performance measurement

    // Iterate over the matrices
    for (const auto &filepath : filepaths)
    {
        SparseMatrixCRS sparseMatrixCRS;         // Matrix in CRS format
        SparseMatrixELLPACK sparseMatrixELLPACK; // Matrix in ELLPACK format

        // Read the matrix from the file and prepare the structures
        readMatrixMarketFile(filepath, sparseMatrixCRS);           // Read the matrix from the file
        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK); // Convert the matrix to ELLPACK format

        // Iterate over the number of columns in the fat vector
        for (const int &k : ks)
        {
            FatVector fatVector, resultCRS, resultELLPACK;            // Fat vector and the results
            generateFatVector(fatVector, sparseMatrixCRS.numCols, k); // Generate the fat vector

            try
            {
                // Perform the matrix-vector multiplication (CRS - Serial)
                resultCRS.values.resize(sparseMatrixCRS.numRows * k, 0.0);                                                  // Resize the result vector
                resultCRS.numCols = k;                                                                                      // Set the number of columns
                resultCRS.numRows = sparseMatrixCRS.numRows;                                                                // Set the number of rows
                double performance_crs = matrixMultivectorProductCRS(sparseMatrixCRS, fatVector, resultCRS, iterationsNum); // Perform the matrix-vector multiplication

                // Perform the matrix-vector multiplication (ELLPACK - Serial)
                resultELLPACK.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);                                                          // Resize the result vector
                resultELLPACK.numCols = k;                                                                                                  // Set the number of columns
                resultELLPACK.numRows = sparseMatrixELLPACK.numRows;                                                                        // Set the number of rows
                double performance_ellpack = matrixMultivectorProductELLPACK(sparseMatrixELLPACK, fatVector, resultELLPACK, iterationsNum); // Perform the matrix-vector multiplication

                // Iterate over the x-axis block size options
                for (const int &xBlockSize : xbdOptions)
                {
                    // Iterate over the y-axis block size options
                    for (const int &yBlockSize : ybdOptions)
                    {
                        FatVector resultCRSCuda, resultELLPACKCuda; // Results

                        // Perform the matrix-vector multiplication (CRS - CUDA)
                        resultCRSCuda.values.resize(sparseMatrixCRS.numRows * k, 0.0);                                                                                   // Resize the result vector
                        resultCRSCuda.numCols = k;                                                                                                                       // Set the number of columns
                        resultCRSCuda.numRows = sparseMatrixCRS.numRows;                                                                                                 // Set the number of rows
                        double performance_crs_cuda = matrixMultivectorProductCRSCUDA(sparseMatrixCRS, fatVector, resultCRSCuda, iterationsNum, xBlockSize, yBlockSize); // Perform the matrix-vector multiplication (CRS - CUDA)

                        // Perform the matrix-vector multiplication (ELLPACK - CUDA)
                        resultELLPACKCuda.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);                                                                                           // Resize the result vector
                        resultELLPACKCuda.numCols = k;                                                                                                                                   // Set the number of columns
                        resultELLPACKCuda.numRows = sparseMatrixELLPACK.numRows;                                                                                                         // Set the number of rows
                        double performance_ellpack_cuda = matrixMultivectorProductELLPACKCUDA(sparseMatrixELLPACK, fatVector, resultELLPACKCuda, iterationsNum, xBlockSize, yBlockSize); // Perform the matrix-vector multiplication (ELLPACK - CUDA)

                        // Compare the results
                        bool crsResultsEqual = areMatricesEqual(resultCRS, resultCRSCuda, 1e-6);             // Compare the results (CRS)
                        bool ellpackResultsEqual = areMatricesEqual(resultELLPACK, resultELLPACKCuda, 1e-6); // Compare the results (ELLPACK)

                        // Print the results
                        int rowsNb = sparseMatrixCRS.numRows;          // Number of rows in the matrix
                        int nonZeroNb = sparseMatrixCRS.values.size(); // Number of non-zero values
                        std::cout << "m: " << rowsNb << ", k: " << k << ", NZ: " << nonZeroNb << ", xBlockSize: " << xBlockSize << ", yBlockSize: " << yBlockSize << ", CRS: " << crsResultsEqual << ", ELLPACK: " << ellpackResultsEqual << ", CRS performance: " << performance_crs_cuda << ", ELLPACK performance: " << performance_ellpack_cuda << ", Serial CRS performance: " << performance_crs << ", Serial ELLPACK performance: " << performance_ellpack << std::endl;

                        // Free the memory
                        resultCRSCuda.values.clear();
                        resultELLPACKCuda.values.clear();
                    }

                    // Free the memory
                    resultCRS.values.clear();
                    resultELLPACK.values.clear();
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "An error occurred: " << e.what() << std::endl;
            }

            // Free the memory
            fatVector.values.clear();
        }

        // Free the memory
        sparseMatrixCRS.values.clear();
        sparseMatrixCRS.colIndices.clear();
        sparseMatrixCRS.rowPtr.clear();
        sparseMatrixELLPACK.values.clear();
        sparseMatrixELLPACK.colIndices.clear();
    }

    return 0;
}