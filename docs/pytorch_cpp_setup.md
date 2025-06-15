# PyTorch C++ Setup Guide

This guide will help you set up PyTorch C++ (LibTorch) for building and running the MAE implementation.

## Prerequisites

- CMake 3.14 or higher
- C++ compiler with C++17 support (GCC 7+, Clang 5+, or MSVC 2017+)
- CUDA 11.x or 12.x (for GPU support)
- OpenCV 4.x

## 1. Download LibTorch

### For CUDA 11.8 (RTX 4090 compatible):
```bash
# Linux
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# macOS (CPU only)
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip

# Windows
# Download from: https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.0%2Bcu118.zip
```

### For CUDA 12.1:
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

## 2. Install OpenCV

### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install libopencv-dev
```

### macOS:
```bash
brew install opencv
```

### Windows:
Download and install from https://opencv.org/releases/

### Build from source (recommended for CUDA support):
```bash
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      ..
make -j$(nproc)
sudo make install
```

## 3. Build the MAE Project

```bash
cd /path/to/MAE
mkdir build && cd build

# Configure with LibTorch path
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
cmake --build . --config Release -j$(nproc)
```

### Common CMake options:
```bash
# Specify CUDA architecture (for RTX 4090)
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DTORCH_CUDA_ARCH_LIST="8.9" \
      ..

# Debug build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch \
      -DCMAKE_BUILD_TYPE=Debug \
      ..
```

## 4. Environment Setup

### Linux:
```bash
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```

### macOS:
```bash
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```

### Windows:
Add `/path/to/libtorch/lib` to your PATH environment variable.

## 5. Verify Installation

Create a simple test program:

```cpp
// test_torch.cpp
#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        torch::Tensor tensor_cuda = tensor.cuda();
        std::cout << tensor_cuda << std::endl;
    } else {
        std::cout << "CUDA not available. Training on CPU." << std::endl;
    }
    
    return 0;
}
```

Build and run:
```bash
g++ -std=c++17 test_torch.cpp -I/path/to/libtorch/include \
    -I/path/to/libtorch/include/torch/csrc/api/include \
    -L/path/to/libtorch/lib \
    -ltorch -ltorch_cpu -lc10 -o test_torch

./test_torch
```

## 6. Troubleshooting

### Common Issues:

1. **CUDA version mismatch**: Ensure LibTorch CUDA version matches your system CUDA version
   ```bash
   nvcc --version  # Check CUDA version
   nvidia-smi      # Check driver version
   ```

2. **ABI compatibility issues on Linux**: Use the CXX11 ABI version of LibTorch
   ```bash
   # Check your GCC ABI
   gcc -v
   ```

3. **Missing dependencies**: Install required libraries
   ```bash
   # Ubuntu/Debian
   sudo apt install libgomp1 libopenblas-dev
   
   # CentOS/RHEL
   sudo yum install libgomp openblas-devel
   ```

4. **CMake cannot find LibTorch**: Specify the full path
   ```bash
   cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
   ```

## 7. Performance Optimization

For optimal performance on RTX 4090:

1. Enable CUDNN:
   ```cpp
   torch::globalContext().setBenchmarkCuDNN(true);
   ```

2. Use mixed precision training (add to training code):
   ```cpp
   torch::autocast::set_enabled(true);
   ```

3. Set CUDA device:
   ```cpp
   torch::cuda::set_device(0);  // Use first GPU
   ```

## 8. Additional Resources

- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/)
- [PyTorch C++ Examples](https://github.com/pytorch/examples/tree/master/cpp)
- [LibTorch Download Links](https://pytorch.org/get-started/locally/)