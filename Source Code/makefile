CXX = g++
NVCC = nvcc
CXXFLAGS = -fopenmp -O3
NVCCFLAGS = -O3 -arch=sm_60
LDFLAGS = -lcudart

# Executables
EXEC_OPENMP_RUN = runOpenMP
EXEC_CUDA_RUN = runCuda
EXEC_OPENMP_TEST = testOpenMP

# Source files
SRCS_OPENMP_RUN := runOpenMP.cpp utils.cpp CRS/matrixMultivectorProductCRS.cpp CRS/matrixMultivectorProductCRSOpenMP.cpp ELLPACK/matrixMultivectorProductELLPACK.cpp ELLPACK/matrixMultivectorProductELLPACKOpenMP.cpp
SRCS_CUDA_RUN := runCuda.cpp utils.cpp ELLPACK/matrixMultivectorProductELLPACKCUDA.cu CRS/matrixMultivectorProductCRS.cpp ELLPACK/matrixMultivectorProductELLPACK.cpp CRS/matrixMultivectorProductCRSCUDA.cu
SRCS_OPENMP_TEST := testOpenMP.cpp utils.cpp CRS/matrixMultivectorProductCRS.cpp CRS/matrixMultivectorProductCRSOpenMP.cpp ELLPACK/matrixMultivectorProductELLPACK.cpp ELLPACK/matrixMultivectorProductELLPACKOpenMP.cpp

# Object files
OBJS_OPENMP_RUN := $(SRCS_OPENMP_RUN:.cpp=.o)
OBJS_CUDA_RUN := $(SRCS_CUDA_RUN:.cpp=.o) $(SRCS_CUDA_RUN:.cu=.o)
OBJS_OPENMP_TEST := $(SRCS_OPENMP_TEST:.cpp=.o)

# Default build
all: $(EXEC_OPENMP_RUN) $(EXEC_CUDA_RUN) $(EXEC_OPENMP_TEST)

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# OpenMP run executable
$(EXEC_OPENMP_RUN): $(OBJS_OPENMP_RUN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# CUDA run executable
$(EXEC_CUDA_RUN): $(OBJS_CUDA_RUN)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

# OpenMP test executable
$(EXEC_OPENMP_TEST): $(OBJS_OPENMP_TEST)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


