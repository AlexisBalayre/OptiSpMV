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

    std::vector<int> ks = {1, 2, 3, 6};

    std::vector<int> xbdOptions = {8, 16, 32, 64};
    std::vector<int> ybdOptions = {1, 2, 4, 8, 16};

    int iterationsNum = 20;

    for (const auto &filepath : filepaths)
    {
        SparseMatrixCRS sparseMatrixCRS;
        SparseMatrixELLPACK sparseMatrixELLPACK;

        // Read the matrix from the file and prepare the structures
        readMatrixMarketFile(filepath, sparseMatrixCRS);
        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK);

        for (const int &k : ks)
        {
            FatVector fatVector, resultCRS, resultELLPACK;
            generateFatVector(fatVector, sparseMatrixCRS.numCols, k);

            try
            {
                // Perform the matrix-vector multiplication (CRS - Serial)
                resultCRS.values.resize(sparseMatrixCRS.numRows * k, 0.0);
                resultCRS.numCols = k;
                resultCRS.numRows = sparseMatrixCRS.numRows;
                double performance_crs = matrixMultivectorProductCRS(sparseMatrixCRS, fatVector, resultCRS, iterationsNum);

                // Perform the matrix-vector multiplication (ELLPACK - Serial)
                resultELLPACK.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
                resultELLPACK.numCols = k;
                resultELLPACK.numRows = sparseMatrixELLPACK.numRows;
                double performance_ellpack = matrixMultivectorProductELLPACK(sparseMatrixELLPACK, fatVector, resultELLPACK, iterationsNum);

                for (const int &xBlockSize : xbdOptions)
                {
                    for (const int &yBlockSize : ybdOptions)
                    {
                        FatVector resultCRSCuda, resultELLPACKCuda;

                        // Perform the matrix-vector multiplication (CRS - CUDA)
                        resultCRSCuda.values.resize(sparseMatrixCRS.numRows * k, 0.0);
                        resultCRSCuda.numCols = k;
                        resultCRSCuda.numRows = sparseMatrixCRS.numRows;
                        double performance_crs_cuda = matrixMultivectorProductCRSCUDA(sparseMatrixCRS, fatVector, resultCRSCuda, iterationsNum, xBlockSize, yBlockSize);

                        // Perform the matrix-vector multiplication (ELLPACK - CUDA)
                        resultELLPACKCuda.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
                        resultELLPACKCuda.numCols = k;
                        resultELLPACKCuda.numRows = sparseMatrixELLPACK.numRows;
                        double performance_ellpack_cuda = matrixMultivectorProductELLPACKCUDA(sparseMatrixELLPACK, fatVector, resultELLPACKCuda, iterationsNum, xBlockSize, yBlockSize);

                        // Compare the results
                        bool crsResultsEqual = areMatricesEqual(resultCRS, resultCRSCuda, 1e-6);
                        bool ellpackResultsEqual = areMatricesEqual(resultELLPACK, resultELLPACKCuda, 1e-6);

                        // Print the results
                        int rowsNb = sparseMatrixCRS.numRows;
                        int nonZeroNb = sparseMatrixCRS.values.size();

                        std::cout << "m: " << rowsNb << ", k: " << k << ", NZ: " << nonZeroNb << ", xBlockSize: " << xBlockSize << ", yBlockSize: " << yBlockSize << ", CRS: " << crsResultsEqual << ", ELLPACK: " << ellpackResultsEqual << ", CRS performance: " << performance_crs_cuda << ", ELLPACK performance: " << performance_ellpack_cuda << ", Serial CRS performance: " << performance_crs << ", Serial ELLPACK performance: " << performance_ellpack << std::endl;

                        // Free the memory
                        resultCRSCuda.values.clear();
                        resultELLPACKCuda.values.clear();
                    }

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