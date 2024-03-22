# OptiSpMV: High-Performance Sparse Matrix-Vector Multiplication

## Overview

OptiSpMV is a pioneering project designed to optimize the multiplication of sparse matrices by fat vectors, a crucial operation in scientific and engineering computations. Leveraging the power of High-Performance Computing (HPC), OptiSpMV employs parallel computing strategies through OpenMP and CUDA, aiming to significantly enhance computational efficiency across different sparse matrix formats: Compressed Sparse Row (CSR) and ELLPACK.

## Project Structure

```graphql
OptiSpMV/
├── CSR/
│   ├── matrixMultivectorProductCRS.* # Source code for the serial implementation of the CSR format
│   ├── matrixMultivectorProductCRSCUDA.* # Source code for the CUDA implementation of the CSR format
│   └── matrixMultivectorProductCRSOpenMP.* # Source code for the OpenMP implementation of the CSR format
├── ELLPACK/
│   ├── matrixMultivectorProductELLPACK.* # Source code for the serial implementation of the ELLPACK format
│   ├── matrixMultivectorProductELLPACKCUDA.* # Source code for the CUDA implementation of the ELLPACK format
│   └── matrixMultivectorProductELLPACKOpenMP.* # Source code for the OpenMP implementation of the ELLPACK format
├── scripts/
│   ├── *.sh # Scripts for performance analysis
│   └── *.sub # Submission scripts for HPC systems
├── utils/
│   ├── utils.* # Utility functions for matrix operations
│   └── MatrixDefinitions.h # Definitions of matrix structures
├── runCuda.cpp # Main program for the CUDA implementation
├── runOpenMP.cpp # Main program for the OpenMP implementation
└── results/
    └── images/ # Directory for storing performance analysis images
```

## Getting Started

### Prerequisites

- An HPC system with MPI and CUDA installed.
- GCC for compiling OpenMP programs.
- NVIDIA CUDA Toolkit for compiling CUDA programs.

### Installation

1. Clone the repository to your local machine or HPC cluster.

   ```bash
   git clone https://github.com/AlexisBalayre/OptiSpMV.git
   ```

2. Navigate to the project directory:

   ```bash
   cd OptiSpMV
   ```

3. Compile the source code. A `makefile` is provided for convenience:

   ```bash
   make all
   ```

### Running the Program

- To run the OpenMP implementation:

  ```bash
  ./runOpenMP
  ```

- To run the CUDA implementation:

  ```bash
  ./runCuda
  ```
