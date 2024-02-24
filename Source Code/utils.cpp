#include "utils.h"

/**
 * @brief Read a matrix from a Matrix Market file
 *
 * @param filename Filepath to the Matrix Market file
 * @param matrix Sparse matrix in CRS format (output)
 */
void readMatrixMarketFile(const std::string &filename, SparseMatrixCRS &matrix)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    bool isSymmetric = false, isPattern = false;
    while (std::getline(file, line))
    {
        if (line[0] == '%')
        {
            if (line.find("symmetric") != std::string::npos)
            {
                isSymmetric = true;
            }
            if (line.find("pattern") != std::string::npos)
            {
                isPattern = true;
            }
        }
        else
        {
            break; // First non-comment line reached, break out of the loop
        }
    }

    int numRows, numCols, nonZeros;
    std::stringstream(line) >> numRows >> numCols >> nonZeros;
    if (!file)
    {
        throw std::runtime_error("Failed to read matrix dimensions from file: " + filename);
    }

    // Clearing existing data
    matrix.values.clear();
    matrix.colIndices.clear();
    matrix.rowPtr.clear();

    matrix.rowPtr.resize(numRows + 1, 0);
    std::vector<std::vector<std::pair<int, double>>> tempRows(numRows);

    int rowIndex, colIndex;
    double value;
    for (int i = 0; i < nonZeros; ++i)
    {
        if (isPattern)
        {
            file >> rowIndex >> colIndex;
            value = 1.0; // Default value for pattern entries
        }
        else
        {
            file >> rowIndex >> colIndex >> value;
        }

        if (!file)
        {
            throw std::runtime_error("Failed to read data from file: " + filename);
        }

        rowIndex--; // Adjusting from 1-based to 0-based indexing
        colIndex--;

        tempRows[rowIndex].emplace_back(colIndex, value);

        if (isSymmetric && rowIndex != colIndex)
        {
            tempRows[colIndex].emplace_back(rowIndex, value);
        }
    }

    // Sort each row by column index
    for (auto &row : tempRows)
    {
        std::sort(row.begin(), row.end());
    }

    // Reconstruct SparseMatrixCRS structure
    int cumSum = 0;
    for (int i = 0; i < numRows; ++i)
    {
        matrix.rowPtr[i] = cumSum;
        for (const auto &elem : tempRows[i])
        {
            matrix.values.push_back(elem.second);
            matrix.colIndices.push_back(elem.first);
        }
        cumSum += tempRows[i].size();
    }
    matrix.rowPtr[numRows] = cumSum;

    matrix.numRows = numRows;
    matrix.numCols = numCols;
}

/**
 * @brief Perform the matrix-vector multiplication for a CRS matrix
 *
 * @param crsMatrix Sparse matrix in CRS format
 * @param ellpackMatrix Pointer to the ELLPACK matrix (output)
 */
void convertCRStoELLPACK(const SparseMatrixCRS &crsMatrix, SparseMatrixELLPACK &ellpackMatrix)
{
    // Calculate the average non-zeros per row to estimate initial allocation
    int totalNonZeros = crsMatrix.values.size();
    int avgNonZerosPerRow = totalNonZeros / crsMatrix.numRows;

    // Initialize with average non-zeros per row to optimize memory usage
    ellpackMatrix.numRows = crsMatrix.numRows;
    ellpackMatrix.numCols = crsMatrix.numCols;
    ellpackMatrix.maxNonZerosPerRow = avgNonZerosPerRow;
    ellpackMatrix.values.resize(ellpackMatrix.numRows * avgNonZerosPerRow);
    ellpackMatrix.colIndices.resize(ellpackMatrix.numRows * avgNonZerosPerRow);

    for (int row = 0; row < crsMatrix.numRows; ++row)
    {
        int rowStart = crsMatrix.rowPtr[row];
        int rowEnd = crsMatrix.rowPtr[row + 1];
        int nonZerosInRow = rowEnd - rowStart;

        // For rows with non-zeros less than the average, this reduces wasted space
        for (int j = rowStart, idx = 0; j < rowEnd; ++j, ++idx)
        {
            int flatIndex = row * avgNonZerosPerRow + idx;
            if (idx < avgNonZerosPerRow)
            {
                ellpackMatrix.values[flatIndex] = crsMatrix.values[j];
                ellpackMatrix.colIndices[flatIndex] = crsMatrix.colIndices[j];
            }
        }
    }
}

/**
 * Method to compare two matrices
 * @param mat1 First matrix
 * @param mat2 Second matrix
 * @param tolerance Tolerance for comparison
 * @return bool True if the matrices are equal, false otherwise
 */
bool areMatricesEqual(const FatVector &mat1, const FatVector &mat2, double tolerance)
{
    if (mat1.numRows != mat2.numRows || mat1.numCols != mat2.numCols)
    {
        return false;
    }

    for (size_t i = 0; i < mat1.values.size(); ++i)
    {
        if (fabs(mat1.values[i] - mat2.values[i]) > tolerance)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief Generate a fat vector
 *
 * @param fatVector Pointer to the fat vector
 * @param numRows Number of rows in the fat vector
 * @param numCols Number of columns in the fat vector
 */
void generateFatVector(FatVector &fatVector, int numRows, int numCols)
{
    fatVector.numRows = numRows;
    fatVector.numCols = numCols;
    fatVector.values.resize(numRows * numCols);

    for (size_t i = 0; i < fatVector.values.size(); ++i)
    {
        fatVector.values[i] = static_cast<double>(rand()) / RAND_MAX; // Valeurs aleatoires entre 0 et 1
    }
}
