#include "utils.h"
#include "MatrixDefinitions.h"
#include "CRS/matrixMultivectorProductCRS.h"
#include "CRS/matrixMultivectorProductCRSOpenMP.h"
#include "ELLPACK/matrixMultivectorProductELLPACK.h"
#include "ELLPACK/matrixMultivectorProductELLPACKOpenMP.h"
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

    int iterationsNum = 20;
    int maxThreadsNum = 16; // Number of threads to use

    std::vector<int> chunkSizes = {2, 4, 8, 16, 32, 64, 128, 256};

    for (const auto &filepath : filepaths)
    {
        SparseMatrixCRS sparseMatrixCRS;
        SparseMatrixELLPACK sparseMatrixELLPACK;

        // Read the matrix from the file and prepare the structures
        readMatrixMarketFile(filepath, sparseMatrixCRS);
        convertCRStoELLPACK(sparseMatrixCRS, sparseMatrixELLPACK);

        for (const int &k : ks)
        {
            // Generate the FatVector
            FatVector fatVector;
            generateFatVector(fatVector, sparseMatrixCRS.numRows, k);

            for (int threads = 1; threads <= maxThreadsNum; threads++)
            {
                try
                {
                    FatVector resultCRS, resultELLPACK;

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

                    for (const int &chunkSize : chunkSizes)
                    {
                        FatVector resultCRSOpenMP, resultELLPACKOpenMP;

                        // Perform the matrix-vector multiplication (CRS - OpenMP)
                        resultCRSOpenMP.values.resize(sparseMatrixCRS.numRows * k, 0.0);
                        resultCRSOpenMP.numCols = k;
                        resultCRSOpenMP.numRows = sparseMatrixCRS.numRows;
                        double performance_crs_openmp = matrixMultivectorProductCRSOpenMP(sparseMatrixCRS, fatVector, resultCRSOpenMP, iterationsNum, threads, chunkSize);

                        // Perform the matrix-vector multiplication (ELLPACK - OpenMP)
                        resultELLPACKOpenMP.values.resize(sparseMatrixELLPACK.numRows * k, 0.0);
                        resultELLPACKOpenMP.numCols = k;
                        resultELLPACKOpenMP.numRows = sparseMatrixELLPACK.numRows;
                        double performance_ellpack_openmp = matrixMultivectorProductELLPACKOpenMP(sparseMatrixELLPACK, fatVector, resultELLPACKOpenMP, iterationsNum, threads, chunkSize);

                        // Compare the results
                        bool crsResultsEqual = areMatricesEqual(resultCRS, resultCRSOpenMP, 1e-6);
                        bool ellpackResultsEqual = areMatricesEqual(resultELLPACK, resultELLPACKOpenMP, 1e-6);

                        // Print the results
                        int rowsNb = sparseMatrixCRS.numRows;
                        int nonZeroNb = sparseMatrixCRS.values.size();

                        std::cout << "m: " << rowsNb << ", k: " << k << ", NZ: " << nonZeroNb << ", threads: " << threads << ", chunk size: " << chunkSize << ", CRS: " << crsResultsEqual << ", ELLPACK: " << ellpackResultsEqual << ", CRS performance: " << performance_crs_openmp << ", ELLPACK performance: " << performance_ellpack_openmp << ", Serial CRS performance: " << performance_crs << ", Serial ELLPACK performance: " << performance_ellpack << std::endl;

                        // Free the memory
                        resultCRSOpenMP.values.clear();
                        resultELLPACKOpenMP.values.clear();
                    }

                    // Free the memory
                    resultCRS.values.clear();
                    resultELLPACK.values.clear();
                }
                catch (const std::exception &e)
                {
                    std::cerr << "An error occurred: " << e.what() << std::endl;
                }
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